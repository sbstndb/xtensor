# Audit C++20 pour xtensor

## Introduction

Ce document présente un audit approfondi des fonctionnalités C++20 qui pourraient être utilisées dans la codebase xtensor. Pour chaque fonctionnalité, nous fournissons :
- **Exemples concrets** tirés du code actuel
- **Avantages** de l'adoption
- **Implications** et effort requis
- **Recommandations** de priorité

Le code actuel utilise déjà C++20 comme standard minimum, mais n'exploite pas encore toutes les fonctionnalités disponibles.

---

## 1. Concepts C++20 - Remplacer SFINAE et std::enable_if

### État actuel

**23 fichiers** utilisent encore `std::enable_if` pour le SFINAE. Le code utilise déjà quelques concepts basiques mais de manière limitée.

### Exemples concrets du code actuel

#### Dans `include/xtensor/core/xexpression.hpp:186-212`

```cpp
// Pattern actuel répété partout
template <class E, class R = void>
using enable_xexpression = typename std::enable_if<is_xexpression<E>::value, R>::type;

template <class E, class R = void>
using disable_xexpression = typename std::enable_if<!is_xexpression<E>::value, R>::type;

template <class E, class R = void>
using enable_xsharable_expression = typename std::enable_if<is_xsharable_expression<E>::value, R>::type;

template <class E, class R = void>
using disable_xsharable_expression = typename std::enable_if<!is_xsharable_expression<E>::value, R>::type;

template <class LHS, class RHS, class R = void>
using enable_assignable_expression = typename std::enable_if<can_assign<LHS, RHS>::value, R>::type;
```

#### Dans `include/xtensor/misc/xtl_concepts.hpp:18-29`

```cpp
// Concepts simples déjà définis
template <typename T>
concept integral_concept = xtl::is_integral<T>::value;

template <typename T>
concept non_integral_concept = !xtl::is_integral<T>::value;

template <typename T>
concept complex_concept = xtl::is_complex<typename std::decay<T>::type::value_type>::value;

template <typename T>
concept pointer_concept = std::is_pointer<T>::value;
```

#### Dans `include/xtensor/utils/xutils.hpp:110`

```cpp
template <class T, class R>
using disable_integral_t = std::enable_if_t<!xtl::is_integral<T>::value, R>;
```

#### Dans `include/xtensor/core/xshape.hpp:144`

```cpp
template <class E, class S, class = typename std::enable_if_t<has_iterator_interface<S>::value>>
inline bool has_shape(const E& e, const S& shape)
```

### Proposition avec concepts C++20

```cpp
// Dans include/xtensor/core/xexpression.hpp
template <class E>
concept xexpression_type = is_xexpression<E>::value;

template <class E>
concept xsharable_expression_type = is_xsharable_expression<E>::value;

template <class LHS, class RHS>
concept assignable_expression = can_assign<LHS, RHS>::value;

// Dans include/xtensor/utils/xutils.hpp
template <typename T>
concept non_integral = !xtl::is_integral<T>::value;

// Dans include/xtensor/core/xshape.hpp
template <class E, class S>
    requires has_iterator_interface<S>::value
inline bool has_shape(const E& e, const S& shape)
{
    return e.shape().size() == shape.size()
           && std::equal(e.shape().cbegin(), e.shape().cend(), shape.begin());
}

// Ou avec syntaxe abrégée:
inline bool has_shape(auto const& e, std::ranges::range auto const& shape)
{
    return e.shape().size() == std::ranges::size(shape)
           && std::ranges::equal(e.shape(), shape);
}
```

### Avantages

1. **Lisibilité** : Messages d'erreur beaucoup plus clairs
   - Avant : `error: no matching function for call to 'foo(bar&)' note: candidate template ignored: disabled by 'enable_if' [with T = int]`
   - Après : `error: no matching function for call to 'foo(bar&)' note: constraints not satisfied because 'bar' does not satisfy 'xexpression_type'`

2. **Performance de compilation** : Les concepts sont évalués plus tôt dans le processus de résolution de surcharge, réduisant les instanciations inutiles

3. **Maintenabilité** : Code plus simple et auto-documenté
   ```cpp
   // Plus clair :
   void process(xexpression_type auto& expr)

   // Que :
   template <class E, enable_xexpression<E>* = nullptr>
   void process(E& expr)
   ```

4. **Composabilité** : Les concepts peuvent être combinés avec `&&`, `||`, `!`
   ```cpp
   template <typename T>
   concept numeric_xexpression = xexpression_type<T> && std::is_arithmetic_v<typename T::value_type>;
   ```

### Implications et effort requis

**Effort** : Moyen-élevé (3-4 semaines)
- ~23 fichiers à refactoriser
- Nécessite une conception soignée de la hiérarchie de concepts
- Tests de régression obligatoires

**Migration progressive possible** :
1. Définir les concepts principaux en parallèle des `enable_if` existants
2. Utiliser les concepts dans le nouveau code
3. Migrer progressivement les anciennes fonctions
4. Déprécier puis supprimer les `enable_if`

**Risques** :
- Changements d'API si les concepts sont trop restrictifs ou trop permissifs
- Peut révéler des bugs cachés par SFINAE

**Compatibilité** :
- Nécessite C++20 minimum (déjà requis)
- Tous les compilateurs supportés doivent supporter les concepts (GCC 10+, Clang 10+, MSVC 2019 16.3+)

### Recommandation

**Priorité : HAUTE** ⭐⭐⭐

Cette migration apporterait le plus grand bénéfice en termes de maintenabilité et d'expérience développeur. À commencer dès que possible par les concepts de base :
1. `xexpression_type`, `xsharable_expression_type`
2. `numeric_type`, `integral_type`, `floating_point_type`
3. Concepts spécifiques aux conteneurs : `xarray_type`, `xtensor_type`

---

## 2. std::span - Vues sur données contiguës

### État actuel

Aucune utilisation de `std::span` détectée. Le code utilise des patterns avec pointeurs + taille ou itérateurs begin/end.

### Exemples de patterns actuels

#### Dans les fonctions utilisant `.data()` et `.size()`

```cpp
// Pattern typique actuel
template <class E>
void process_contiguous(E& expr)
{
    auto* ptr = expr.data();
    auto size = expr.size();

    for (std::size_t i = 0; i < size; ++i) {
        // traiter ptr[i]
    }
}

// Dans xshape.hpp:112-114
template <class S1, class S2>
inline bool same_shape(const S1& s1, const S2& s2) noexcept
{
    return s1.size() == s2.size() && std::equal(s1.begin(), s1.end(), s2.begin());
}
```

### Proposition avec std::span

```cpp
template <class T>
void process_contiguous(std::span<T> data)
{
    for (auto& elem : data) {
        // traiter elem
    }
}

// Ou avec concepts:
void process_contiguous(std::ranges::contiguous_range auto& expr)
{
    std::span view{expr};
    // ...
}

// Pour same_shape:
inline bool same_shape(std::span<const std::size_t> s1, std::span<const std::size_t> s2) noexcept
{
    return std::ranges::equal(s1, s2);
}
```

### Avantages

1. **Sécurité** : Span porte sa taille, évitant les erreurs de désynchronisation pointeur/taille
   ```cpp
   // Dangereux :
   void foo(int* data, size_t size);
   foo(arr.data(), wrong_size); // Compile mais erreur !

   // Sûr :
   void foo(std::span<int> data);
   foo(arr); // Taille déduite automatiquement
   ```

2. **Interopérabilité** : Interface standard pour travailler avec différents types de conteneurs
   ```cpp
   void process(std::span<const double> data);

   std::vector<double> vec = {...};
   std::array<double, 10> arr = {...};
   double c_arr[5] = {...};

   process(vec);   // OK
   process(arr);   // OK
   process(c_arr); // OK
   ```

3. **Sous-vues** : Création facile de sous-vues sans copie
   ```cpp
   std::span<int> data = get_data();
   auto first_half = data.subspan(0, data.size() / 2);
   auto second_half = data.subspan(data.size() / 2);
   ```

4. **Meilleure optimisation** : Le compilateur peut mieux optimiser avec `std::span` qu'avec des pointeurs nus

### Implications et effort requis

**Effort** : Faible-moyen (1-2 semaines)
- Identifier les fonctions qui prendraient `std::span` en paramètre
- Ajouter des surcharges ou remplacer les signatures existantes
- Surtout bénéfique pour les nouvelles fonctions

**Cas d'usage dans xtensor** :
1. Interface avec des buffers externes (I/O, sérialisation)
2. Fonctions de manipulation de shape/strides
3. Algorithmes SIMD sur données contiguës
4. Fonctions utilitaires prenant des ranges de tailles connues

**Limitations** :
- Seulement pour les données **contiguës** en mémoire
- Pas adapté pour les tensors avec strides non-triviaux
- Pas un remplacement universel des itérateurs xtensor

### Recommandation

**Priorité : MOYENNE** ⭐⭐

Utile principalement pour :
- Nouvelles APIs publiques prenant des buffers de données
- Fonctions utilitaires sur shapes/strides/indices
- Interfaces avec le code externe

Ne pas remplacer aveuglément tous les pointeurs, seulement là où cela apporte une valeur ajoutée.

---

## 3. std::ranges - Bibliothèque Ranges

### État actuel

Aucune utilisation de `std::ranges`. Le code utilise les algorithmes `<algorithm>` classiques avec itérateurs.

### Exemples de patterns actuels

#### Dans `include/xtensor/core/xshape.hpp:112-114`

```cpp
template <class S1, class S2>
inline bool same_shape(const S1& s1, const S2& s2) noexcept
{
    return s1.size() == s2.size() && std::equal(s1.begin(), s1.end(), s2.begin());
}
```

#### Pattern typique dans tout le code

```cpp
// Boucles manuelles qu'on trouve partout
for (std::size_t i = 0; i < shape.size(); ++i) {
    result[i] = shape[i] * stride[i];
}

// Ou avec itérateurs
std::transform(s1.begin(), s1.end(), s2.begin(), result.begin(), std::multiplies<>{});
```

### Proposition avec std::ranges

```cpp
// same_shape avec ranges
template <std::ranges::sized_range S1, std::ranges::sized_range S2>
inline bool same_shape(const S1& s1, const S2& s2) noexcept
{
    return std::ranges::equal(s1, s2);
}

// Ou encore plus simple si on veut juste comparer :
inline bool same_shape(auto const& s1, auto const& s2) noexcept
{
    return std::ranges::equal(s1, s2);
}

// Transformations fonctionnelles avec ranges
auto compute_strides(auto const& shape, auto const& stride) {
    return std::views::zip_transform(std::multiplies{}, shape, stride)
         | std::ranges::to<std::vector>();
}

// Vues paresseuses (lazy evaluation)
auto get_positive_elements(auto const& container) {
    return container
         | std::views::filter([](auto x) { return x > 0; })
         | std::views::transform([](auto x) { return x * 2; });
}
```

### Avantages

1. **Composabilité** : Chaîner des opérations de manière élégante
   ```cpp
   // Sans ranges : plusieurs boucles ou temporaires
   std::vector<int> temp;
   std::copy_if(data.begin(), data.end(), std::back_inserter(temp),
                [](int x) { return x > 0; });
   std::vector<int> result;
   std::transform(temp.begin(), temp.end(), std::back_inserter(result),
                  [](int x) { return x * 2; });

   // Avec ranges : une seule expression, pas de temporaires
   auto result = data
               | std::views::filter([](int x) { return x > 0; })
               | std::views::transform([](int x) { return x * 2; })
               | std::ranges::to<std::vector>();
   ```

2. **Évaluation paresseuse** : Les vues ne copient pas, calculs à la demande
   ```cpp
   auto squares = std::views::iota(0, 1000000)
                | std::views::transform([](int x) { return x * x; });

   // Aucun calcul effectué ici ^

   auto first_10 = squares | std::views::take(10);
   // Seuls 10 carrés sont calculés, pas 1 million !
   ```

3. **Lisibilité** : Code plus déclaratif que impératif
   ```cpp
   // Lisible : dit QUOI faire
   auto odds = data | std::views::filter([](int x) { return x % 2; });

   // Vs impératif : dit COMMENT faire
   std::vector<int> odds;
   for (auto x : data) {
       if (x % 2) odds.push_back(x);
   }
   ```

4. **Performance** : Évite les allocations intermédiaires, meilleure optimisation

5. **Intégration naturelle avec les itérateurs xtensor** : xtensor pourrait exposer des vues ranges pour ses itérateurs

### Cas d'usage spécifiques à xtensor

```cpp
// 1. Manipulation de shapes
auto broadcasted_shape = std::views::zip_transform(
    [](auto s1, auto s2) { return std::max(s1, s2); },
    shape1, shape2
) | std::ranges::to<dynamic_shape<std::size_t>>();

// 2. Génération de séquences d'indices
auto indices = std::views::iota(0uz, tensor.size())
             | std::views::transform([&](auto i) {
                   return tensor.linear_begin()[i];
               });

// 3. Filtrage de tensors
auto positive_values = tensor
                     | std::views::filter([](auto x) { return x > 0; });

// 4. Transformations paresseuses pour expression templates
// (complémente l'existant, ne remplace pas)
```

### Implications et effort requis

**Effort** : Moyen (2-3 semaines)
- Identifier les algorithmes qui bénéficieraient des ranges
- Ajouter des overloads acceptant ranges
- Documenter les nouveaux patterns

**Points d'attention** :
- Ne pas casser l'API existante
- Les ranges complètent les expression templates, ne les remplacent pas
- Attention à la compatibilité des itérateurs xtensor avec `std::ranges::range`
- Certains itérateurs xtensor peuvent nécessiter des adaptations pour satisfaire les concepts ranges

**Opportunités** :
1. Nouvelles fonctions utilitaires utilisant ranges
2. APIs alternatives pour certaines opérations
3. Exemples dans la documentation montrant les patterns ranges

### Recommandation

**Priorité : MOYENNE-BASSE** ⭐⭐

Utile pour :
- Nouveau code utilitaire (manipulation de shapes, indices)
- APIs alternatives pour les utilisateurs préférant le style ranges
- Documentation et exemples

Ne pas refactoriser massivement le code existant, car :
- xtensor a déjà un excellent système d'expression templates
- Les ranges ne remplacent pas les expression templates pour les tensors
- Risque de complexifier sans bénéfice majeur

Adopter progressivement dans le nouveau code et les exemples.

---

## 4. consteval - Fonctions compile-time garanties

### État actuel

Aucune utilisation de `consteval`. Le code utilise `constexpr` massivement, mais sans garantie que l'évaluation se fasse à la compilation.

### Exemples de patterns actuels

#### Dans `include/xtensor/core/xmath.hpp:38-53`

```cpp
template <class T = double>
struct numeric_constants
{
    static constexpr T PI = 3.141592653589793238463;
    static constexpr T PI_2 = 1.57079632679489661923;
    static constexpr T PI_4 = 0.785398163397448309616;
    static constexpr T E = 2.71828182845904523536;
    // ...
};
```

#### Dans `include/xtensor/core/xmath.hpp:84-99`

```cpp
#define XTENSOR_UNARY_MATH_FUNCTOR(NAME)              \
    struct NAME##_fun                                 \
    {                                                 \
        template <class T>                            \
        constexpr auto operator()(const T& arg) const \
        {                                             \
            using math::NAME;                         \
            return NAME(arg);                         \
        }                                             \
    }
```

#### Dans `include/xtensor/utils/xutils.hpp:54-55`

```cpp
template <std::size_t I, class... Args>
constexpr decltype(auto) argument(Args&&... args) noexcept;
```

### Proposition avec consteval

```cpp
// 1. Calculs de shape au compile-time
template <std::size_t... Dims>
consteval std::size_t compute_total_size() {
    return (Dims * ...);
}

template <std::size_t... Dims>
consteval auto make_compile_time_shape() {
    return std::array{Dims...};
}

// Usage :
constexpr auto shape = make_compile_time_shape<3, 4, 5>();
constexpr auto size = compute_total_size<3, 4, 5>(); // 60

// 2. Vérifications de validité au compile-time
template <std::size_t... Dims>
consteval bool are_dims_valid() {
    return ((Dims > 0) && ...);
}

template <std::size_t... Dims>
    requires are_dims_valid<Dims...>()
class static_tensor {
    // ...
};

// 3. Construction d'indices au compile-time
template <std::size_t N>
consteval auto make_index_sequence_array() {
    std::array<std::size_t, N> result{};
    for (std::size_t i = 0; i < N; ++i) {
        result[i] = i;
    }
    return result;
}

// 4. Calculs de layout au compile-time
enum class layout_type { row_major, column_major, dynamic };

template <std::size_t... Dims>
consteval auto compute_row_major_strides() {
    constexpr std::size_t N = sizeof...(Dims);
    std::array<std::size_t, N> dims{Dims...};
    std::array<std::size_t, N> strides{};

    strides[N - 1] = 1;
    for (std::size_t i = N - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// 5. Validation de paramètres template
template <typename T>
consteval bool is_valid_value_type() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>>
        || std::is_same_v<T, std::complex<double>>;
}

template <typename T>
    requires is_valid_value_type<T>()
class xarray {
    // ...
};
```

### Avantages

1. **Garantie compile-time** : Erreur si la fonction ne peut pas être évaluée au compile-time
   ```cpp
   constexpr int runtime_value() { return rand(); } // Compile
   consteval int compiletime_value() { return rand(); } // Erreur !
   ```

2. **Optimisation** : Code plus rapide car calculs faits à la compilation
   ```cpp
   // Sans consteval : calculé à chaque appel runtime
   constexpr auto strides = compute_strides(shape);

   // Avec consteval : calculé une seule fois à la compilation
   consteval auto strides = compute_strides(shape);
   // strides est dans le binaire, pas de calcul runtime !
   ```

3. **Validation statique** : Vérifier la validité des paramètres avant l'exécution
   ```cpp
   template <std::size_t... Dims>
       requires (consteval { return are_dims_positive<Dims...>(); }())
   class tensor { /* ... */ };

   tensor<3, 4, 5> ok;     // OK
   tensor<3, 0, 5> error;  // Erreur de compilation !
   ```

4. **Documentation** : `consteval` documente l'intention (cette fonction DOIT être évaluée au compile-time)

### Implications et effort requis

**Effort** : Faible-moyen (1-2 semaines)
- Identifier les fonctions qui sont toujours appelées avec des constantes
- Ajouter `consteval` aux fonctions de calcul de métadonnées
- Créer des helpers `consteval` pour la validation

**Cas d'usage dans xtensor** :
1. Calculs de taille/shape/strides pour `fixed_shape`
2. Validation de paramètres template
3. Construction de tables de lookup au compile-time
4. Métaprogrammation (sélection de types, calculs sur types)

**Précautions** :
- Ne fonctionne que si **tous** les arguments sont connus au compile-time
- Ne pas abuser : `constexpr` est suffisant si la fonction peut être appelée au runtime aussi
- Peut ralentir la compilation si trop de calculs lourds

**Règle d'usage** :
```cpp
// Utiliser consteval si :
// 1. La fonction est toujours appelée avec des constantes compile-time
// 2. Un échec à évaluer au compile-time devrait être une erreur

// Utiliser constexpr si :
// 1. La fonction peut être appelée au runtime aussi
// 2. C'est une optimisation opportuniste
```

### Recommandation

**Priorité : FAIBLE-MOYENNE** ⭐⭐

Utile pour :
- Fonctions de calcul de métadonnées pour `fixed_shape` et types statiques
- Validation de paramètres template
- Amélioration de la sémantique du code (documentation d'intention)

Adopter progressivement :
1. Identifier les fonctions `constexpr` toujours appelées avec des constantes
2. Créer des versions `consteval` pour les nouveaux helpers compile-time
3. Documenter les patterns recommandés

Pas de refactoring massif nécessaire, mais bénéfique pour le nouveau code.

---

## 5. Opérateur Spaceship (<=>) - Comparaisons trois-voies

### État actuel

Aucune utilisation de `operator<=>`. Le code implémente probablement `operator==`, `operator!=`, `operator<`, etc. séparément.

### Pattern actuel typique

```cpp
// Pattern classique C++17
template <class D>
class xexpression {
public:
    bool operator==(const xexpression& other) const {
        return derived_cast() == other.derived_cast();
    }

    bool operator!=(const xexpression& other) const {
        return !(*this == other);
    }

    bool operator<(const xexpression& other) const {
        // comparaison element-wise...
    }

    bool operator<=(const xexpression& other) const {
        return !(other < *this);
    }

    bool operator>(const xexpression& other) const {
        return other < *this;
    }

    bool operator>=(const xexpression& other) const {
        return !(*this < other);
    }
};
// 6 fonctions pour gérer toutes les comparaisons !
```

### Proposition avec operator<=>

```cpp
// Version C++20 avec spaceship
template <class D>
class xexpression {
public:
    // Option 1 : Défaut (génère ==, !=, <, <=, >, >=)
    auto operator<=>(const xexpression& other) const = default;

    // Option 2 : Implémentation custom
    std::strong_ordering operator<=>(const xexpression& other) const {
        auto& lhs = derived_cast();
        auto& rhs = other.derived_cast();

        if (lhs.size() != rhs.size()) {
            return lhs.size() <=> rhs.size();
        }

        return std::lexicographical_compare_three_way(
            lhs.begin(), lhs.end(),
            rhs.begin(), rhs.end()
        );
    }

    // == peut être généré automatiquement ou défini séparément si besoin
    bool operator==(const xexpression& other) const = default;
};

// Pour les shapes
template <class S1, class S2>
auto compare_shapes(const S1& s1, const S2& s2) {
    return std::lexicographical_compare_three_way(
        s1.begin(), s1.end(),
        s2.begin(), s2.end()
    );
}

// Avec concepts pour les comparaisons
template <typename T>
concept totally_ordered_value = std::totally_ordered<T>;

template <totally_ordered_value T>
auto compare_arrays(std::span<const T> a, std::span<const T> b) {
    return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
}
```

### Types de comparaison

C++20 offre trois catégories de comparaison :

```cpp
// 1. strong_ordering : types avec ordre total (int, float, string, etc.)
//    a == b  =>  !(a < b) && !(b < a)
//    Valeurs : less, equal, greater
std::strong_ordering cmp = 5 <=> 10; // strong_ordering::less

// 2. weak_ordering : ordre partiel (insensible à la casse par exemple)
//    Valeurs : less, equivalent, greater
std::weak_ordering cmp = "hello"_ci <=> "HELLO"_ci; // equivalent

// 3. partial_ordering : ordre partiel avec NaN (float, double)
//    Valeurs : less, equivalent, greater, unordered
std::partial_ordering cmp = 1.0 <=> NaN; // partial_ordering::unordered
```

### Avantages

1. **Moins de code** : Une seule fonction au lieu de 6
   ```cpp
   // Avant : ~30 lignes
   bool operator==(const T&) const;
   bool operator!=(const T&) const;
   bool operator<(const T&) const;
   bool operator<=(const T&) const;
   bool operator>(const T&) const;
   bool operator>=(const T&) const;

   // Après : ~3 lignes
   auto operator<=>(const T&) const = default;
   bool operator==(const T&) const = default;
   ```

2. **Moins d'erreurs** : Impossible d'avoir des incohérences entre `<` et `>=`
   ```cpp
   // Avant : risque d'incohérence
   bool operator<(const T& other) const { return size() < other.size(); }
   bool operator>=(const T& other) const { return size() > other.size(); } // Bug !

   // Après : cohérence garantie
   auto operator<=>(const T& other) const { return size() <=> other.size(); }
   ```

3. **Performance** : Une seule comparaison au lieu de plusieurs
   ```cpp
   // Avant : deux comparaisons pour trier
   if (a < b) { /* ... */ }
   else if (b < a) { /* ... */ }
   else { /* égaux */ }

   // Après : une seule comparaison
   switch (a <=> b) {
       case std::strong_ordering::less:    /* ... */
       case std::strong_ordering::greater: /* ... */
       case std::strong_ordering::equal:   /* ... */
   }
   ```

4. **Meilleure intégration** : Types compatibles avec `std::sort`, `std::set`, etc. automatiquement

### Implications et effort requis

**Effort** : Moyen (2-3 semaines)
- Identifier les classes avec opérateurs de comparaison
- Remplacer par `operator<=>` et `operator==`
- Tester la cohérence sémantique (surtout pour les tensors)

**Points d'attention pour xtensor** :

1. **Comparaisons élément-par-élément** : xtensor fait probablement des comparaisons élément-par-élément retournant un tensor de bool
   ```cpp
   // xtensor actuel (supposé)
   auto result = arr1 < arr2; // retourne un xarray<bool>

   // <=> retourne un ordre, pas un tensor
   auto order = arr1 <=> arr2; // strong_ordering, pas xarray<bool> !
   ```

2. **Ne pas casser l'API existante** : Les comparaisons élément-par-élément sont essentielles
   ```cpp
   // Solution : garder les deux
   class xarray {
       // Comparaison élément-par-élément (existant)
       friend auto operator<(const xarray& a, const xarray& b) {
           return element_wise_less(a, b); // retourne xarray<bool>
       }

       // Comparaison lexicographique (nouveau)
       std::strong_ordering compare_to(const xarray& other) const {
           return std::lexicographical_compare_three_way(
               linear_begin(), linear_end(),
               other.linear_begin(), other.linear_end()
           );
       }
   };
   ```

3. **Sémantique de comparaison** : Qu'est-ce que `arr1 < arr2` devrait signifier ?
   - Comparaison élément-par-élément → `xarray<bool>`
   - Comparaison lexicographique → `bool`
   - Toutes les valeurs de `arr1` < correspondantes de `arr2` → `bool`

### Recommandation

**Priorité : FAIBLE** ⭐

Raisons :
1. Les comparaisons de tensors sont sémantiquement complexes (élément-par-élément vs lexicographique)
2. Risque de confusion avec l'API existante
3. Bénéfice limité pour xtensor comparé à d'autres features

**Usage recommandé** :
- Classes utilitaires (shapes, indices, strides) : `operator<=>` OK
- Classes de tensors : garder l'API actuelle, peut-être ajouter `.compare_to()` séparément
- Nouveaux types simples : utiliser `operator<=>` par défaut

---

## 6. Abbreviated Function Templates - Templates abrégés

### État actuel

Le code utilise la syntaxe classique `template <class T>` partout. 43 occurrences de `auto func(...) ->` avec trailing return type.

### Exemples de patterns actuels

```cpp
// Pattern typique partout dans le code
template <class E1, class E2>
void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);

template <class F, class... Args>
decltype(auto) apply(F&& func, Args&&... args);

template <std::size_t I, class... Args>
constexpr decltype(auto) argument(Args&&... args) noexcept;

template <class E>
auto derived_cast() & noexcept -> derived_type&;
```

### Proposition avec templates abrégés

```cpp
// Syntaxe simple
void assign_data(auto& e1, const auto& e2, bool trivial);

decltype(auto) apply(auto&& func, auto&&... args);

constexpr decltype(auto) argument(auto&&... args) noexcept;

auto derived_cast() & noexcept -> derived_type&;

// Avec concepts (meilleur)
void assign_data(xexpression_type auto& e1, const xexpression_type auto& e2, bool trivial);

decltype(auto) apply(std::invocable auto&& func, auto&&... args);

// Lambdas génériques (déjà possible C++14, amélioré C++20)
auto transform_shape = []<typename T>(std::span<T> data) {
    return std::views::transform(data, [](T x) { return x * 2; });
};

// Template lambdas (nouveau C++20)
auto generic_print = []<typename... Args>(Args&&... args) {
    (std::cout << ... << std::forward<Args>(args));
};
```

### Avantages

1. **Concision** : Moins de bruit syntaxique
   ```cpp
   // Avant : 63 caractères
   template <class T, class U>
   void foo(T&& t, U&& u)

   // Après : 31 caractères
   void foo(auto&& t, auto&& u)
   ```

2. **Lisibilité** : Plus proche du code non-template
   ```cpp
   // Ressemble à du code normal
   auto process(auto const& input) {
       return transform(input);
   }
   ```

3. **Lambdas template** : Nouveauté C++20 très utile
   ```cpp
   // Impossible en C++17
   auto generic_lambda = []<typename T>(std::vector<T> const& vec) {
       // T est accessible dans le corps de la lambda !
       return sizeof(T) * vec.size();
   };

   // En C++17, il faut passer par un hack
   auto old_lambda = [](auto const& vec) {
       using T = typename std::decay_t<decltype(vec)>::value_type;
       return sizeof(T) * vec.size();
   };
   ```

4. **Concepts intégrés** : Combine naturellement avec les concepts
   ```cpp
   void process(std::integral auto x, std::floating_point auto y) {
       // x est forcément un type entier
       // y est forcément un type flottant
   }
   ```

### Implications et effort requis

**Effort** : Faible (quelques jours)
- Transformation purement syntaxique
- Pas de changement sémantique
- Peut se faire progressivement

**Stratégie de migration** :
1. Nouveau code : utiliser les templates abrégés par défaut
2. Code existant : migrer lors des modifications
3. Exceptions : garder `template<class T>` si le nom du type est important pour la documentation

**Guidelines** :
```cpp
// ✅ Bon : concept explicite
void process(xexpression_type auto& expr)

// ⚠️  Acceptable : auto simple (mais moins documenté)
void process(auto& expr)

// ❌ Éviter : trop verbeux sans bénéfice
template <class E>
void process(E& expr)

// ✅ Exception : le nom du paramètre template est important
template <class ValueType>  // ValueType documente l'intention
auto create_array(std::size_t size) -> xarray<ValueType>
```

### Cas d'usage spécifiques à xtensor

```cpp
// 1. Fonctions utilitaires
auto same_shape(auto const& s1, auto const& s2) noexcept {
    return std::ranges::equal(s1, s2);
}

// 2. Fonctions avec concepts
void assign(xexpression_type auto& lhs, const xexpression_type auto& rhs) {
    // ...
}

// 3. Lambdas template pour métaprogrammation
constexpr auto compute_strides = []<std::size_t... Dims>() {
    return std::array{compute_stride<Dims>()...};
};

// 4. Perfect forwarding simplifié
decltype(auto) forward_to_derived(auto&& expr) {
    return std::forward<decltype(expr)>(expr).derived_cast();
}

// 5. Fonctions génériques avec contraintes
auto reduce(xexpression_type auto const& expr, std::invocable<auto, auto> auto&& op) {
    return std::reduce(expr.begin(), expr.end(), typename decltype(expr)::value_type{}, op);
}
```

### Recommandation

**Priorité : MOYENNE** ⭐⭐

Avantages :
- Migration facile et progressive
- Améliore la lisibilité du nouveau code
- S'intègre bien avec les concepts
- Lambdas template très utiles pour la métaprogrammation

Recommandation :
1. **Adopter immédiatement** pour tout nouveau code
2. Définir des guidelines claires (quand utiliser `auto` vs `concept auto` vs `template<class T>`)
3. Migrer progressivement le code existant lors des modifications
4. Privilégier `concept auto` plutôt que `auto` seul pour la documentation

---

## 7. constexpr amélioré - Plus de constexpr

### État actuel

Le code utilise déjà `constexpr` massivement. C++20 autorise plus de choses en `constexpr` :
- Destructeurs virtuels `constexpr`
- `constexpr dynamic_cast` et `typeid`
- Allocation dynamique `constexpr` (new/delete, std::vector, std::string)
- Fonctions `constexpr` avec `try`/`catch`
- `constexpr std::vector`, `constexpr std::string`

### Exemples de limitations C++17

```cpp
// ❌ Impossible en C++17
constexpr std::vector<int> make_vector(int size) {
    std::vector<int> v(size);  // Erreur : allocation dynamique
    for (int i = 0; i < size; ++i) {
        v[i] = i * i;
    }
    return v;
}

// ❌ Impossible en C++17
constexpr std::string make_string(const char* str) {
    return std::string(str);  // Erreur : allocation dynamique
}

// ❌ Impossible en C++17
class Base {
public:
    constexpr virtual ~Base() = default;  // Erreur : virtual
};
```

### Proposition avec C++20 constexpr

```cpp
// ✅ OK en C++20
constexpr std::vector<int> make_vector(int size) {
    std::vector<int> v(size);
    for (int i = 0; i < size; ++i) {
        v[i] = i * i;
    }
    return v;
}

// ✅ OK en C++20
constexpr std::string make_string(const char* str) {
    return std::string(str);
}

// ✅ OK en C++20
class Base {
public:
    constexpr virtual ~Base() = default;
};

// Exemples pour xtensor

// 1. Création de shapes dynamiques au compile-time
constexpr auto make_dynamic_shape(std::size_t rank) {
    dynamic_shape<std::size_t> shape(rank);
    for (std::size_t i = 0; i < rank; ++i) {
        shape[i] = i + 1;
    }
    return shape;
}

constexpr auto shape = make_dynamic_shape(3); // {1, 2, 3} au compile-time

// 2. Calculs de strides au compile-time
constexpr auto compute_strides(const auto& shape) {
    std::vector<std::size_t> strides(shape.size());
    if (!strides.empty()) {
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    return strides;
}

constexpr std::array shape{3uz, 4uz, 5uz};
constexpr auto strides = compute_strides(shape); // {20, 5, 1} au compile-time !

// 3. Vérification de compatibilité de shapes au compile-time
constexpr bool check_broadcastable(const auto& shape1, const auto& shape2) {
    auto it1 = shape1.rbegin();
    auto it2 = shape2.rbegin();

    while (it1 != shape1.rend() && it2 != shape2.rend()) {
        if (*it1 != *it2 && *it1 != 1 && *it2 != 1) {
            return false;
        }
        ++it1;
        ++it2;
    }
    return true;
}

constexpr std::array s1{3uz, 1uz, 5uz};
constexpr std::array s2{3uz, 4uz, 1uz};
static_assert(check_broadcastable(s1, s2)); // Vérifié à la compilation !

// 4. Construction de tables de lookup au compile-time
constexpr auto make_lookup_table(std::size_t size, auto generator) {
    std::vector<double> table(size);
    for (std::size_t i = 0; i < size; ++i) {
        table[i] = generator(i);
    }
    return table;
}

constexpr auto sin_table = make_lookup_table(360, [](auto deg) {
    return std::sin(deg * 3.14159265359 / 180.0);
});
// Table précalculée dans le binaire !

// 5. Validation de paramètres template au compile-time
template <typename T>
constexpr bool is_valid_numeric_type() {
    try {
        // Même les try/catch sont autorisés en constexpr C++20
        if (!std::is_arithmetic_v<T>) return false;
        if (sizeof(T) > 16) return false;  // Limite arbitraire
        return true;
    } catch (...) {
        return false;
    }
}
```

### Avantages

1. **Plus de calculs au compile-time** : Économies runtime
   ```cpp
   // Table calculée une fois à la compilation, pas à chaque exécution
   constexpr auto coefficients = compute_polynomial_coeffs(degree);
   ```

2. **Validation statique plus puissante**
   ```cpp
   template <auto Shape>
       requires (check_shape_valid(Shape))
   class tensor { /* ... */ };
   ```

3. **Code plus simple** : Pas besoin de séparer constexpr et non-constexpr
   ```cpp
   // Avant : deux versions
   template <typename T>
   constexpr std::array<T, N> compute_static();  // Pour compile-time

   template <typename T>
   std::vector<T> compute_dynamic(int n);  // Pour runtime

   // Après : une seule version
   template <typename T>
   constexpr auto compute(auto size) {  // std::vector OK !
       std::vector<T> result(size);
       // ...
       return result;
   }
   ```

4. **Meilleure intégration avec concepts**
   ```cpp
   template <typename T>
   concept ValidNumericType = is_valid_numeric_type<T>();  // constexpr C++20
   ```

### Implications et effort requis

**Effort** : Faible-moyen (1-2 semaines)
- Identifier les fonctions qui pourraient être constexpr mais ne le sont pas à cause des limitations C++17
- Ajouter `constexpr` là où c'est pertinent
- Créer des tests de constexpr-ness avec `static_assert`

**Opportunités dans xtensor** :

1. **dynamic_shape** avec `constexpr`
   ```cpp
   constexpr dynamic_shape<std::size_t> shape = {3, 4, 5};
   // Utilisable au compile-time !
   ```

2. **Fonctions de calcul de strides/backstrides** avec shapes dynamiques
   ```cpp
   constexpr auto strides = compute_strides(shape, layout::row_major);
   ```

3. **Validation de configurations au compile-time**
   ```cpp
   template <auto Config>
       requires validate_config(Config)  // Config peut contenir std::vector !
   class configured_tensor { /* ... */ };
   ```

4. **Tables de lookup précalculées**
   ```cpp
   constexpr auto trig_table = precompute_trig_values();
   // Intégré dans le binaire, pas de calcul runtime
   ```

**Limitations** :
- L'allocation doit être libérée dans la même évaluation constexpr (pas de leak)
- Toujours impossible de faire des I/O ou appels systèmes en constexpr
- Le code doit rester "pur" (sans effets de bord observables)

### Recommandation

**Priorité : MOYENNE** ⭐⭐

Utile pour :
- Permettre plus de calculs compile-time pour les types statiques (`fixed_shape`, etc.)
- Simplifier le code (une seule implémentation pour compile-time et runtime)
- Validation plus puissante au compile-time

Adoption :
1. Ajouter `constexpr` aux fonctions existantes qui en bénéficieraient
2. Créer des tests `static_assert` pour vérifier la constexpr-ness
3. Documenter quelles fonctions sont évaluables au compile-time

Bénéfices modérés mais effort faible, donc recommandé.

---

## 8. Designated Initializers - Initialiseurs nommés

### État actuel

Pas d'utilisation de designated initializers (nouveauté C++20, inspiré de C99).

### Pattern actuel

```cpp
// Structs de configuration typiques
struct tensor_config {
    layout_type layout;
    bool enable_simd;
    std::size_t alignment;
    bool bounds_checking;
};

// Initialisation positionnelle : ordre important, pas clair
tensor_config cfg{layout_type::row_major, true, 64, false};
// Quel paramètre est quoi ? 🤔
```

### Proposition avec designated initializers

```cpp
// Même struct
struct tensor_config {
    layout_type layout;
    bool enable_simd;
    std::size_t alignment;
    bool bounds_checking;
};

// Initialisation nommée : ordre flexible, auto-documenté
tensor_config cfg{
    .layout = layout_type::row_major,
    .enable_simd = true,
    .alignment = 64,
    .bounds_checking = false
};

// Ordre différent : OK
tensor_config cfg2{
    .bounds_checking = true,
    .layout = layout_type::column_major,
    .alignment = 32,
    .enable_simd = false
};

// Certains membres omis : initialisés par défaut
tensor_config minimal{
    .layout = layout_type::row_major
    // enable_simd, alignment, bounds_checking : valeurs par défaut
};
```

### Exemples pour xtensor

```cpp
// 1. Configuration de création de tensor
struct array_params {
    layout_type layout = layout_type::row_major;
    std::size_t alignment = 0;
    bool initialize = true;
};

auto arr = xarray<double>({3, 4, 5}, array_params{
    .layout = layout_type::column_major,
    .initialize = false
});

// 2. Options de reduction
struct reduce_options {
    std::size_t axis = 0;
    bool keep_dims = false;
    evaluation_strategy::type strategy = evaluation_strategy::lazy;
};

auto result = reduce(tensor, reduce_options{
    .axis = 1,
    .keep_dims = true
});

// 3. Options d'I/O
struct save_options {
    bool compress = false;
    std::string_view format = "npy";
    int compression_level = 6;
};

save("data.npy", tensor, save_options{
    .compress = true,
    .compression_level = 9
});

// 4. Configuration de slicing
struct slice_params {
    std::size_t start = 0;
    std::optional<std::size_t> stop = std::nullopt;
    std::ptrdiff_t step = 1;
};

auto view = strided_view(tensor, {
    slice_params{.start = 0, .stop = 10, .step = 2},
    slice_params{.start = 5}
});
```

### Avantages

1. **Auto-documentation** : Le code explique ce qu'il fait
   ```cpp
   // Avant : obscur
   create_tensor({3, 4}, row_major, true, 64, false);

   // Après : clair
   create_tensor({3, 4}, {
       .layout = row_major,
       .enable_simd = true,
       .alignment = 64,
       .bounds_check = false
   });
   ```

2. **Sûreté** : Ordre des membres n'importe pas
   ```cpp
   // Si on ajoute un membre au milieu de la struct :
   struct config {
       int a;
       int new_member;  // Ajouté ici
       int b;
   };

   // Avant : mauvais comportement silencieux
   config c{1, 2};  // a=1, new_member=2, b=0 (pas ce qu'on voulait !)

   // Après : toujours correct
   config c{.a = 1, .b = 2};  // new_member = 0 (défaut)
   ```

3. **Valeurs par défaut** : Membres omis = valeurs par défaut
   ```cpp
   struct options {
       bool verbose = false;
       int level = 1;
       std::string name = "default";
   };

   // Spécifier seulement ce qui change
   options opt{.verbose = true};
   // level = 1, name = "default" automatiquement
   ```

4. **Lisibilité** : Surtout pour structs avec beaucoup de membres
   ```cpp
   struct rendering_params {
       float ambient, diffuse, specular, shininess;
       color background, foreground;
       bool shadows, reflections, antialiasing;
       // ... 20 autres paramètres
   };

   // Avant : 😱
   rendering_params params{0.3, 0.7, 0.5, 32.0, {0,0,0}, {255,255,255}, true, false, true, /* ... */};

   // Après : 😊
   rendering_params params{
       .ambient = 0.3,
       .diffuse = 0.7,
       .specular = 0.5,
       .shininess = 32.0,
       .background = {0, 0, 0},
       .foreground = {255, 255, 255},
       .shadows = true,
       .antialiasing = true
       // Autres = défaut
   };
   ```

### Limitations C++20

```cpp
// ⚠️  Ordre doit correspondre à la déclaration (contrairement à C99)
struct s { int a, b, c; };

s obj1{.a = 1, .c = 3};           // ✅ OK (b omis)
s obj2{.a = 1, .b = 2, .c = 3};   // ✅ OK
s obj3{.c = 3, .a = 1};           // ❌ Erreur : ordre incorrect

// ⚠️  Pas pour les tableaux (contrairement à C99)
int arr[3] = {[0] = 1, [2] = 3};  // ❌ Erreur en C++

// ⚠️  Pas de mélange positionnel/nommé
s obj4{1, .b = 2};                // ❌ Erreur

// ⚠️  Pas pour les classes avec constructeurs non-triviaux
class C {
public:
    C(int x) : value(x) {}
    int value;
};
C obj{.value = 10};               // ❌ Erreur
```

### Implications et effort requis

**Effort** : Très faible (quelques heures à 1 jour)
- Identifier les structs de configuration
- Ajouter des valeurs par défaut aux membres
- Mettre à jour la documentation et exemples

**Quand utiliser** :
✅ Structs de configuration/options
✅ Structs POD avec plusieurs membres
✅ Paramètres de fonctions complexes

❌ Pas pour classes avec invariants
❌ Pas si ordre a du sens sémantique
❌ Pas pour structs avec peu de membres (1-2)

**Cas d'usage xtensor** :
1. Options de création de tensors
2. Paramètres d'algorithmes (reduce, sort, etc.)
3. Configuration d'I/O
4. Options de visualisation/printing
5. Paramètres de slicing/indexing

### Recommandation

**Priorité : FAIBLE-MOYENNE** ⭐⭐

Avantages :
- Très facile à adopter
- Améliore nettement la lisibilité
- Réduit les erreurs
- Pas de coût performance

Recommandation :
1. Utiliser systématiquement pour tout nouveau code avec structs d'options
2. Encourager dans les exemples et documentation
3. Pas besoin de refactorer massivement l'existant
4. Définir des guidelines (quand utiliser vs quand ne pas utiliser)

---

## 9. Modules C++20 (Avancé)

### État actuel

xtensor utilise le système de headers classique (.hpp). Aucun module.

### Concept des modules

```cpp
// Avant : headers
// xarray.hpp
#ifndef XTENSOR_XARRAY_HPP
#define XTENSOR_XARRAY_HPP

#include <vector>
#include <memory>
// ... 50 autres includes

namespace xt {
    template <class T>
    class xarray {
        // ...
    };
}

#endif

// Après : modules
// xarray.cppm
export module xt.xarray;

import <vector>;
import <memory>;
// ... 50 autres imports

export namespace xt {
    template <class T>
    class xarray {
        // ...
    };
}

// Utilisation
import xt.xarray;

auto arr = xt::xarray<double>{};
```

### Avantages théoriques

1. **Temps de compilation** : Drastiquement réduit (10x-100x possible)
   - Headers parsés une seule fois, pas à chaque TU
   - Pas de dépendances transitives implicites
   - Parallélisation meilleure

2. **Ordre d'inclusion** : N'a plus d'importance
   ```cpp
   // Headers : ordre important
   #include <xtensor/xarray.hpp>
   #include <xtensor/xview.hpp>  // Peut échouer si ordre incorrect

   // Modules : ordre sans importance
   import xt.xarray;
   import xt.xview;  // Toujours OK
   ```

3. **Isolation** : Pas de pollution de namespace
   ```cpp
   // Headers : macros fuient
   #define MAX(a, b) ((a) > (b) ? (a) : (b))
   #include <user_code.hpp>  // Voit MAX 😱

   // Modules : isolation complète
   export module foo;
   #define MAX(a, b) ((a) > (b) ? (a) : (b))
   // MAX n'est PAS exporté, pas de fuite
   ```

4. **Encapsulation** : Contrôle fin de ce qui est exporté
   ```cpp
   module xt.internal;

   namespace xt::detail {
       void helper() { /* ... */ }  // Pas exporté
   }

   export namespace xt {
       void public_api() { /* ... */ }  // Exporté
   }
   ```

### Problèmes et limitations (2025)

1. **Support compilateur** : Encore incomplet et incompatible
   - GCC, Clang, MSVC : implémentations différentes
   - Bugs et incompatibilités fréquents
   - Support outils (CMake, build systems) immature

2. **Migration massive** : Effort colossal pour xtensor
   - 74 headers à convertir
   - Dépendances externes (xtl, xsimd) doivent aussi migrer
   - Tests exhaustifs nécessaires

3. **Bibliothèques header-only** : Moins de bénéfices
   - xtensor est header-only
   - Modules excellent pour binaires, moins pour header-only

4. **Compatibilité** : Impossible de mixer facilement
   - Code utilisateur devrait migrer aussi
   - Période de transition longue et douloureuse

5. **Écosystème** : Standard library pas encore modulée partout
   ```cpp
   import std;  // Devrait importer toute la stdlib
   // Mais : support incomplet, comportement variable selon compilateurs
   ```

### Recommandation

**Priorité : TRÈS FAIBLE** ❌

**NE PAS MIGRER maintenant** pour les raisons suivantes :

1. **Immaturité** : Technologie pas encore prête en production (2025)
2. **Effort/bénéfice** : Ratio très défavorable pour une lib header-only
3. **Écosystème** : Dépendances (xtl, xsimd) doivent migrer d'abord
4. **Utilisateurs** : Forcerait la migration de tout code utilisateur
5. **Outils** : Build systems, IDE, debuggers encore limités

**Quand reconsidérer** (≥ 2027-2028) :
- ✅ Support compilateur universel et stable
- ✅ `import std;` fonctionnel partout
- ✅ CMake et outils de build supportent bien
- ✅ Dépendances (xtl, xsimd) migrées
- ✅ Retours d'expérience positifs de la communauté

**En attendant** :
- Surveiller l'évolution de la technologie
- Expérimenter dans des branches/projets tests
- Préparer mentalement la migration future
- Structurer le code pour faciliter une éventuelle migration

---

## 10. Autres fonctionnalités C++20

### 10.1 - Template lambdas avec paramètres template

**Vu dans section 6**, mais mérite d'être souligné.

```cpp
// Nouveau C++20 : paramètres template explicites dans lambdas
auto size_of_elements = []<typename T>(const std::vector<T>& vec) {
    return sizeof(T) * vec.size();
};

// Utile pour xtensor
auto broadcast_shape = []<typename... Shapes>(const Shapes&... shapes) {
    return compute_broadcast_shape(shapes...);
};

// Lambdas constexpr avec template
constexpr auto factorial = []<std::size_t N>() consteval {
    if constexpr (N == 0) return 1;
    else return N * factorial.template operator()<N-1>();
};

constexpr auto fact5 = factorial.template operator()<5>();  // 120 au compile-time
```

**Priorité : MOYENNE** ⭐⭐ - Très utile pour la métaprogrammation.

---

### 10.2 - using enum (C++20)

```cpp
enum class layout_type { row_major, column_major, dynamic };

// Avant C++20
void foo(layout_type layout) {
    switch (layout) {
        case layout_type::row_major: break;
        case layout_type::column_major: break;
        case layout_type::dynamic: break;
    }
}

// C++20
void foo(layout_type layout) {
    using enum layout_type;  // Importe tous les enum
    switch (layout) {
        case row_major: break;
        case column_major: break;
        case dynamic: break;
    }
}

// Ou import sélectif
using layout_type::row_major;
auto default_layout = row_major;  // Au lieu de layout_type::row_major
```

**Priorité : FAIBLE** ⭐ - Commodité syntaxique mineure.

---

### 10.3 - [[likely]] / [[unlikely]] attributes

```cpp
// Aide l'optimiseur
bool check_bounds(size_t index, size_t size) {
    if (index < size) [[likely]] {  // Cas le plus probable
        return true;
    } else [[unlikely]] {  // Cas rare
        throw std::out_of_range("Index out of bounds");
    }
}

// Pour xtensor
template <class E>
auto& xcontainer::operator[](size_t i) {
    #ifdef XTENSOR_ENABLE_ASSERT
    if (i >= size()) [[unlikely]] {
        throw_index_error(i);
    }
    #endif
    return data()[i];
}

// Switch avec likelihood
switch (layout) {
    [[likely]] case layout_type::row_major:
        return row_major_access(index);
    case layout_type::column_major:
        return column_major_access(index);
    [[unlikely]] case layout_type::dynamic:
        return dynamic_access(index);
}
```

**Priorité : FAIBLE-MOYENNE** ⭐⭐ - Optimisation micro, mais peut aider dans hot paths.

---

### 10.4 - [[no_unique_address]] attribute

```cpp
// Optimisation de taille pour membres vides (Empty Base Optimization généralisé)
struct empty_t {};

// Avant
struct S {
    empty_t e;  // Prend 1 byte même si vide !
    int value;  // sizeof(S) = 8 (padding)
};

// Après
struct S {
    [[no_unique_address]] empty_t e;  // Prend 0 bytes !
    int value;  // sizeof(S) = 4
};

// Pour xtensor : optimiser les types avec allocator/tag vides
template <class T, class Allocator = std::allocator<T>>
class xarray {
    [[no_unique_address]] Allocator alloc_;  // 0 bytes si allocator vide
    std::size_t size_;
    T* data_;
};
```

**Priorité : FAIBLE-MOYENNE** ⭐⭐ - Optimisation mémoire, surtout pour classes avec beaucoup de tags/policies.

---

### 10.5 - std::source_location (remplacement de __FILE__/__LINE__)

```cpp
#include <source_location>

// Avant
void log_error(const char* msg, const char* file = __FILE__, int line = __LINE__) {
    std::cerr << file << ":" << line << ": " << msg << "\n";
}

// Après
void log_error(const char* msg,
               std::source_location loc = std::source_location::current()) {
    std::cerr << loc.file_name() << ":" << loc.line()
              << " in " << loc.function_name() << ": " << msg << "\n";
}

// Pour xtensor : améliorer les messages d'erreur
void throw_shape_error(const auto& expected, const auto& actual,
                       std::source_location loc = std::source_location::current()) {
    throw std::runtime_error(
        std::format("{}:{} in {}: Shape mismatch: expected {:n}, got {:n}",
                    loc.file_name(), loc.line(), loc.function_name(),
                    expected, actual)
    );
}

// Usage
if (!same_shape(a, b)) {
    throw_shape_error(a.shape(), b.shape());  // Capture automatiquement file/line/function
}
```

**Priorité : MOYENNE** ⭐⭐ - Très utile pour le débogage et messages d'erreur.

---

### 10.6 - std::format (ou std::print en C++23)

⚠️ **Note** : `std::format` est C++20 mais pas encore disponible dans tous les compilateurs en 2025. Vérifier la compatibilité.

```cpp
#include <format>

// Avant
std::string msg = "Tensor shape: [" + std::to_string(dim0) + ", "
                + std::to_string(dim1) + ", " + std::to_string(dim2) + "]";

// Après
auto msg = std::format("Tensor shape: [{}, {}, {}]", dim0, dim1, dim2);

// Formatage avancé
auto detailed = std::format(
    "Tensor<{}>: shape={:n}, size={}, layout={}",
    type_name<T>(),
    shape,
    size,
    layout_name(layout)
);

// Custom formatter pour xtensor types
template <>
struct std::formatter<xt::xarray<double>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const xt::xarray<double>& arr, format_context& ctx) const {
        return std::format_to(ctx.out(),
            "xarray<double>(shape={}, size={})",
            arr.shape(), arr.size());
    }
};

// Usage
xt::xarray<double> arr = {{1, 2}, {3, 4}};
std::cout << std::format("Array: {}\n", arr);
// Output: Array: xarray<double>(shape=[2, 2], size=4)
```

**Priorité : MOYENNE-HAUTE** ⭐⭐⭐ - Très utile pour logging, débogage, messages d'erreur. **MAIS** vérifier compatibilité compilateurs.

---

## Résumé des priorités

| Fonctionnalité | Priorité | Effort | Impact | Recommandation |
|----------------|----------|--------|--------|----------------|
| **1. Concepts** | ⭐⭐⭐ Haute | Moyen-élevé | Très élevé | **Commencer dès que possible** |
| **2. std::span** | ⭐⭐ Moyenne | Faible-moyen | Moyen | Adopter pour nouveau code |
| **3. std::ranges** | ⭐⭐ Moyenne-basse | Moyen | Moyen | Adopter progressivement |
| **4. consteval** | ⭐⭐ Faible-moyenne | Faible-moyen | Moyen | Utiliser pour nouveau code compile-time |
| **5. Spaceship <=>** | ⭐ Faible | Moyen | Faible | Seulement pour types utilitaires |
| **6. Templates abrégés** | ⭐⭐ Moyenne | Très faible | Moyen-élevé | **Adopter immédiatement** |
| **7. constexpr amélioré** | ⭐⭐ Moyenne | Faible-moyen | Moyen | Ajouter où pertinent |
| **8. Designated init** | ⭐⭐ Faible-moyenne | Très faible | Moyen | Utiliser pour structs config |
| **9. Modules** | ❌ Très faible | Très élevé | Incertain | **NE PAS MIGRER** (trop tôt) |
| **10.1 Template lambdas** | ⭐⭐ Moyenne | Faible | Moyen | Utile pour métaprogrammation |
| **10.2 using enum** | ⭐ Faible | Très faible | Faible | Commodité mineure |
| **10.3 [[likely]]** | ⭐⭐ Faible-moyenne | Très faible | Faible-moyen | Hot paths seulement |
| **10.4 [[no_unique_address]]** | ⭐⭐ Faible-moyenne | Très faible | Faible-moyen | Optimisation mémoire |
| **10.5 source_location** | ⭐⭐ Moyenne | Faible | Moyen | Améliore messages erreur |
| **10.6 std::format** | ⭐⭐⭐ Moyenne-haute | Faible-moyen | Élevé | Très utile MAIS vérifier compat |

---

## Plan de migration recommandé

### Phase 1 : Quick Wins (1-2 mois)
1. ✅ **Templates abrégés** : Adopter dans tout nouveau code
2. ✅ **Designated initializers** : Pour structs de configuration
3. ✅ **[[no_unique_address]]** : Pour optimiser taille des classes
4. ✅ **std::source_location** : Améliorer messages d'erreur

### Phase 2 : Fondations (3-6 mois)
1. ✅ **Concepts de base** : Définir hiérarchie de concepts
   - `xexpression_type`, `numeric_type`, `container_type`, etc.
2. ✅ **Migration SFINAE → Concepts** : Progressivement
   - Commencer par `xexpression.hpp`
   - Puis `xutils.hpp`, `xshape.hpp`
3. ✅ **constexpr amélioré** : Fonctions de calcul shape/strides
4. ✅ **consteval** : Helpers compile-time pour fixed_shape

### Phase 3 : Optimisations (6-12 mois)
1. ✅ **std::span** : APIs prenant buffers/ranges
2. ✅ **std::ranges** : Fonctions utilitaires, exemples
3. ✅ **[[likely]]/[[unlikely]]** : Hot paths identifiés par profiling
4. ✅ **std::format** : Si compatibilité suffisante

### Phase 4 : Futur (>12 mois ou plus tard)
1. ⏸️ **Spaceship operator** : Évaluer bénéfices réels
2. ⏸️ **Modules** : Attendre maturité écosystème (2027-2028?)

---

## Conclusion

xtensor est bien positionné pour adopter les fonctionnalités C++20. Les priorités principales sont :

1. **Concepts** : Le plus grand impact sur maintenabilité et expérience développeur
2. **Templates abrégés** : Facile à adopter, améliore lisibilité immédiatement
3. **Designated initializers** : Simple et améliore l'API utilisateur
4. **std::source_location** : Meilleurs messages d'erreur sans effort

Les fonctionnalités à adopter progressivement :
- **std::span**, **constexpr/consteval**, **std::ranges**

Les fonctionnalités à éviter pour l'instant :
- **Modules** (trop tôt)
- **Spaceship** (peu de bénéfices pour xtensor)

La migration peut se faire **progressivement** sans casser la compatibilité, en introduisant les nouvelles fonctionnalités au fur et à mesure.

---

**Document créé le 2025-10-21**
**Basé sur l'analyse de xtensor 0.27.1**
