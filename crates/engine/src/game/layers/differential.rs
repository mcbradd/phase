//! Differential verification of the entry-incremental layer flush.
//!
//! CR 613.1: continuous effects are evaluated in layer order over the whole
//! board. `flush_layers`' `EnteredObjects` arm re-derives only the freshly
//! entered objects (plus any host they attached to) and leaves every
//! pre-existing permanent's already-derived characteristics untouched. That is
//! sound only while the escalation gate in `prepare_incremental_flush` proves
//! the shortcut yields a board identical to a full pass.
//!
//! This module verifies that premise instead of asserting it: when enabled, the
//! post-entry board is cloned BEFORE the incremental pass mutates it, the clone
//! is re-derived with a full `evaluate_layers`, and the two boards are compared
//! field-by-field over EVERY battlefield object — not just the entered ones,
//! since the interesting failure mode is precisely a pre-existing object the
//! full pass updates and the incremental pass does not.
//!
//! Comparing the whole board is also what gives the CR 613.8a dependency- and
//! timestamp-ordering axis real coverage: `Abilities` / `StaticDefinitions` /
//! `Keywords` are compared as ordered sequences, so a hybrid board on which the
//! incremental arm applies two effects in the wrong relative order diverges
//! here rather than silently shipping a differently-ordered board.
//!
//! Cost and gating. Cloning a `GameState` and running a full evaluation is far
//! more expensive than the pass being checked, so this is compiled only under
//! `cfg(test)` or the `differential-flush` feature, and is additionally OFF at
//! runtime unless the feature is on (or a test switches it on for its own
//! thread). Disabled, `scratch_before_flush` returns `None` and nothing is
//! cloned.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;

use crate::types::game_state::GameState;
use crate::types::identifiers::ObjectId;
use crate::types::zones::Zone;

/// The comparison classes. One variant per field of [`RecipientDerived`], plus
/// [`DivergenceClass::Presence`] for an object that exists on one board and not
/// the other.
///
/// Compile-time tripwire #1: `label` matches exhaustively with no `_` arm, so a
/// new class cannot be added without being named in the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DivergenceClass {
    Presence,
    Name,
    Power,
    Toughness,
    Loyalty,
    CardTypes,
    ManaCost,
    Keywords,
    Abilities,
    ReplacementDefinitions,
    StaticDefinitions,
    Color,
    PrintedRef,
    DisplaySource,
    TokenImageRef,
    Controller,
    AssignsDamageFromToughness,
    AssignsDamageAsThoughUnblocked,
    AssignsNoCombatDamage,
}

pub(crate) const CLASS_COUNT: usize = 19;

/// The harness's inspection surface (this constant plus [`set_enabled`],
/// [`comparison_counts`], [`reset_comparison_counts`]) is consumed by this
/// module's own unit tests. It stays compiled in the feature-only build —
/// `pub(crate)` puts it out of reach of the integration-test crates — so the
/// non-vacuity contract lives in one place rather than being conditionally
/// present depending on how the harness was built.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const ALL_CLASSES: [DivergenceClass; CLASS_COUNT] = [
    DivergenceClass::Presence,
    DivergenceClass::Name,
    DivergenceClass::Power,
    DivergenceClass::Toughness,
    DivergenceClass::Loyalty,
    DivergenceClass::CardTypes,
    DivergenceClass::ManaCost,
    DivergenceClass::Keywords,
    DivergenceClass::Abilities,
    DivergenceClass::ReplacementDefinitions,
    DivergenceClass::StaticDefinitions,
    DivergenceClass::Color,
    DivergenceClass::PrintedRef,
    DivergenceClass::DisplaySource,
    DivergenceClass::TokenImageRef,
    DivergenceClass::Controller,
    DivergenceClass::AssignsDamageFromToughness,
    DivergenceClass::AssignsDamageAsThoughUnblocked,
    DivergenceClass::AssignsNoCombatDamage,
];

impl DivergenceClass {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Presence => "presence",
            Self::Name => "name",
            Self::Power => "power",
            Self::Toughness => "toughness",
            Self::Loyalty => "loyalty",
            Self::CardTypes => "card_types",
            Self::ManaCost => "mana_cost",
            Self::Keywords => "keywords",
            Self::Abilities => "abilities",
            Self::ReplacementDefinitions => "replacement_definitions",
            Self::StaticDefinitions => "static_definitions",
            Self::Color => "color",
            Self::PrintedRef => "printed_ref",
            Self::DisplaySource => "display_source",
            Self::TokenImageRef => "token_image_ref",
            Self::Controller => "controller",
            Self::AssignsDamageFromToughness => "assigns_damage_from_toughness",
            Self::AssignsDamageAsThoughUnblocked => "assigns_damage_as_though_unblocked",
            Self::AssignsNoCombatDamage => "assigns_no_combat_damage",
        }
    }
}

/// Every derived field the incremental recipient reset
/// (`layers::reset_recipient_to_base`) restores from its base counterpart,
/// captured as a comparable snapshot.
///
/// Compile-time tripwire #2: [`compare_snapshots`] destructures this struct with
/// NO `..` rest pattern on either side, so adding a field to the reset — and
/// therefore to this struct — fails to compile until the field is also compared
/// and given a [`DivergenceClass`]. Composite characteristics are rendered
/// through `Debug` rather than compared with `PartialEq`, both because
/// `GameObject`'s characteristic types do not uniformly implement it and
/// because the rendering is order-sensitive, which is what makes the CR 613.8a
/// ordering axis observable.
///
/// `tests::every_reset_field_is_observed_by_the_comparator` is the runtime half
/// of the tie: for each field the production reset writes, it asserts that
/// perturbing the live value moves this snapshot, so a reset field that never
/// reaches this struct is caught by test rather than by a wrong board in
/// production.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecipientDerived {
    name: String,
    power: Option<i32>,
    toughness: Option<i32>,
    loyalty: Option<u32>,
    card_types: String,
    mana_cost: String,
    keywords: String,
    abilities: String,
    replacement_definitions: String,
    static_definitions: String,
    color: String,
    printed_ref: String,
    display_source: String,
    token_image_ref: String,
    controller: String,
    assigns_damage_from_toughness: bool,
    assigns_damage_as_though_unblocked: bool,
    assigns_no_combat_damage: bool,
}

impl RecipientDerived {
    pub(crate) fn of(obj: &crate::game::game_object::GameObject) -> Self {
        Self {
            name: obj.name.clone(),
            power: obj.power,
            toughness: obj.toughness,
            loyalty: obj.loyalty,
            card_types: format!("{:?}", obj.card_types),
            mana_cost: format!("{:?}", obj.mana_cost),
            keywords: format!("{:?}", obj.keywords),
            abilities: format!("{:?}", obj.abilities),
            replacement_definitions: format!("{:?}", obj.replacement_definitions),
            static_definitions: format!("{:?}", obj.static_definitions),
            color: format!("{:?}", obj.color),
            printed_ref: format!("{:?}", obj.printed_ref),
            display_source: format!("{:?}", obj.display_source),
            token_image_ref: format!("{:?}", obj.token_image_ref),
            controller: format!("{:?}", obj.controller),
            assigns_damage_from_toughness: obj.assigns_damage_from_toughness,
            assigns_damage_as_though_unblocked: obj.assigns_damage_as_though_unblocked,
            assigns_no_combat_damage: obj.assigns_no_combat_damage,
        }
    }
}

/// One field-level disagreement between the incremental board and the
/// forced-full reference board.
#[derive(Debug, Clone)]
pub(crate) struct Divergence {
    pub(crate) object: ObjectId,
    pub(crate) class: DivergenceClass,
    pub(crate) incremental: String,
    pub(crate) full: String,
}

#[cfg(feature = "differential-flush")]
const DEFAULT_ENABLED: bool = true;
#[cfg(not(feature = "differential-flush"))]
const DEFAULT_ENABLED: bool = false;

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(DEFAULT_ENABLED) };
    /// Per-class count of comparisons PERFORMED (not divergences found). A
    /// harness test asserts every class is > 0, which is what distinguishes
    /// "the comparator found nothing wrong" from "the comparator never looked".
    static COMPARISONS: RefCell<[u64; CLASS_COUNT]> = const { RefCell::new([0; CLASS_COUNT]) };
}

pub(crate) fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// Enable/disable for the CURRENT THREAD only — `cargo test` runs tests in
/// parallel threads, so a process-wide switch would let one test's setting
/// decide another test's cost and behavior.
///
/// Inspection surface — see [`ALL_CLASSES`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn set_enabled(on: bool) -> bool {
    ENABLED.with(|e| e.replace(on))
}

/// RAII switch used to give the harness BLOCKING coverage in the default
/// `cargo test` path rather than only under the feature.
///
/// Without this the module compiles under `cfg(test)` but is runtime-dead
/// (`DEFAULT_ENABLED` is `false`), so it contributes nothing executable to CI —
/// the failure mode a nightly `continue-on-error` job has as well. Holding
/// [`DifferentialSwitch::on`] around a flush that takes the `EnteredObjects`
/// arm makes the comparison run, and `verify`'s panic makes a divergence a red
/// test.
///
/// There is deliberately no `off` counterpart. One would only be needed by a
/// fixture that drives a board the incremental arm gets wrong on purpose and
/// asserts the divergence itself, and no such fixture remains — every entry
/// flush fixture runs enforced. Note that were one reintroduced it would have to
/// SET `false` rather than merely decline to enable, because under the
/// `differential-flush` feature the thread default is already `true`, so "don't
/// turn it on" is not the same as "turn it off".
///
/// The previous setting is restored on drop, so neither the cost nor the
/// assertion leaks into the next test scheduled on the same thread.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct DifferentialSwitch(bool);

// The constructors are reached only from `#[cfg(test)]` fixtures; under a
// plain `--features differential-flush` build the type is a diagnostic
// affordance with no in-tree caller, so `dead_code` fires on the associated
// functions independently of the attribute on the struct itself.
#[cfg_attr(not(test), allow(dead_code))]
impl DifferentialSwitch {
    pub(crate) fn on() -> Self {
        Self(set_enabled(true))
    }
}

impl Drop for DifferentialSwitch {
    fn drop(&mut self) {
        set_enabled(self.0);
    }
}

/// Inspection surface — see [`ALL_CLASSES`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn comparison_counts() -> [u64; CLASS_COUNT] {
    COMPARISONS.with(|c| *c.borrow())
}

/// Inspection surface — see [`ALL_CLASSES`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn reset_comparison_counts() {
    COMPARISONS.with(|c| *c.borrow_mut() = [0; CLASS_COUNT]);
}

fn record_comparison(class: DivergenceClass) {
    COMPARISONS.with(|c| c.borrow_mut()[class.index()] += 1);
}

/// Clone the post-entry board before the incremental pass mutates it.
///
/// Must be called BEFORE `prepare_incremental_flush`, which resets recipients to
/// their base characteristics as its first act: a clone taken afterwards would
/// compare the incremental result against a board the incremental pass had
/// already half-built.
pub(crate) fn scratch_before_flush(state: &GameState) -> Option<GameState> {
    enabled().then(|| state.clone())
}

/// Compare two boards field-by-field over every battlefield object.
pub(crate) fn compare_boards(incremental: &GameState, full: &GameState) -> Vec<Divergence> {
    let ids: BTreeSet<ObjectId> = incremental
        .objects
        .iter()
        .chain(full.objects.iter())
        .filter(|(_, obj)| obj.zone == Zone::Battlefield)
        .map(|(id, _)| *id)
        .collect();

    let mut divergences = Vec::new();
    for id in ids {
        let a = incremental.objects.get(&id);
        let b = full.objects.get(&id);
        record_comparison(DivergenceClass::Presence);
        match (a, b) {
            (Some(a), Some(b)) => divergences.extend(compare_snapshots(
                id,
                &RecipientDerived::of(a),
                &RecipientDerived::of(b),
            )),
            (a, b) => divergences.push(Divergence {
                object: id,
                class: DivergenceClass::Presence,
                incremental: format!("{}", a.is_some()),
                full: format!("{}", b.is_some()),
            }),
        }
    }
    divergences
}

/// Compile-time tripwire #2 lives here: both snapshots are destructured with no
/// `..`, so every field of [`RecipientDerived`] must be named and compared.
fn compare_snapshots(
    object: ObjectId,
    incremental: &RecipientDerived,
    full: &RecipientDerived,
) -> Vec<Divergence> {
    let RecipientDerived {
        name: a_name,
        power: a_power,
        toughness: a_toughness,
        loyalty: a_loyalty,
        card_types: a_card_types,
        mana_cost: a_mana_cost,
        keywords: a_keywords,
        abilities: a_abilities,
        replacement_definitions: a_replacement_definitions,
        static_definitions: a_static_definitions,
        color: a_color,
        printed_ref: a_printed_ref,
        display_source: a_display_source,
        token_image_ref: a_token_image_ref,
        controller: a_controller,
        assigns_damage_from_toughness: a_assigns_damage_from_toughness,
        assigns_damage_as_though_unblocked: a_assigns_damage_as_though_unblocked,
        assigns_no_combat_damage: a_assigns_no_combat_damage,
    } = incremental;
    let RecipientDerived {
        name: b_name,
        power: b_power,
        toughness: b_toughness,
        loyalty: b_loyalty,
        card_types: b_card_types,
        mana_cost: b_mana_cost,
        keywords: b_keywords,
        abilities: b_abilities,
        replacement_definitions: b_replacement_definitions,
        static_definitions: b_static_definitions,
        color: b_color,
        printed_ref: b_printed_ref,
        display_source: b_display_source,
        token_image_ref: b_token_image_ref,
        controller: b_controller,
        assigns_damage_from_toughness: b_assigns_damage_from_toughness,
        assigns_damage_as_though_unblocked: b_assigns_damage_as_though_unblocked,
        assigns_no_combat_damage: b_assigns_no_combat_damage,
    } = full;

    let mut out = Vec::new();
    let mut check = |class: DivergenceClass, a: String, b: String| {
        record_comparison(class);
        if a != b {
            out.push(Divergence {
                object,
                class,
                incremental: a,
                full: b,
            });
        }
    };
    check(DivergenceClass::Name, a_name.clone(), b_name.clone());
    check(
        DivergenceClass::Power,
        format!("{a_power:?}"),
        format!("{b_power:?}"),
    );
    check(
        DivergenceClass::Toughness,
        format!("{a_toughness:?}"),
        format!("{b_toughness:?}"),
    );
    check(
        DivergenceClass::Loyalty,
        format!("{a_loyalty:?}"),
        format!("{b_loyalty:?}"),
    );
    check(
        DivergenceClass::CardTypes,
        a_card_types.clone(),
        b_card_types.clone(),
    );
    check(
        DivergenceClass::ManaCost,
        a_mana_cost.clone(),
        b_mana_cost.clone(),
    );
    check(
        DivergenceClass::Keywords,
        a_keywords.clone(),
        b_keywords.clone(),
    );
    check(
        DivergenceClass::Abilities,
        a_abilities.clone(),
        b_abilities.clone(),
    );
    check(
        DivergenceClass::ReplacementDefinitions,
        a_replacement_definitions.clone(),
        b_replacement_definitions.clone(),
    );
    check(
        DivergenceClass::StaticDefinitions,
        a_static_definitions.clone(),
        b_static_definitions.clone(),
    );
    check(DivergenceClass::Color, a_color.clone(), b_color.clone());
    check(
        DivergenceClass::PrintedRef,
        a_printed_ref.clone(),
        b_printed_ref.clone(),
    );
    check(
        DivergenceClass::DisplaySource,
        a_display_source.clone(),
        b_display_source.clone(),
    );
    check(
        DivergenceClass::TokenImageRef,
        a_token_image_ref.clone(),
        b_token_image_ref.clone(),
    );
    check(
        DivergenceClass::Controller,
        a_controller.clone(),
        b_controller.clone(),
    );
    check(
        DivergenceClass::AssignsDamageFromToughness,
        a_assigns_damage_from_toughness.to_string(),
        b_assigns_damage_from_toughness.to_string(),
    );
    check(
        DivergenceClass::AssignsDamageAsThoughUnblocked,
        a_assigns_damage_as_though_unblocked.to_string(),
        b_assigns_damage_as_though_unblocked.to_string(),
    );
    check(
        DivergenceClass::AssignsNoCombatDamage,
        a_assigns_no_combat_damage.to_string(),
        b_assigns_no_combat_damage.to_string(),
    );
    out
}

/// Run the forced-full reference pass on `scratch` and compare it against the
/// board the incremental arm just produced. Panics on any divergence: an
/// incremental board that disagrees with a full pass is a wrong board, and the
/// harness exists to stop it at the seed that produced it rather than to let it
/// be averaged into a run.
pub(crate) fn verify(state: &GameState, scratch: Option<GameState>) {
    let Some(mut scratch) = scratch else {
        return;
    };
    // The scratch board is the SAME post-entry board, so a full evaluation of it
    // is the reference the incremental arm claims to be equivalent to. It runs
    // under counter suppression: the shadow pass is a diagnostic, not a pass the
    // engine performed, so it must not be visible to the `cfg(test)` counters
    // that pin how much work the production flush did.
    super::without_counter_observation(|| super::evaluate_layers(&mut scratch));
    let divergences = compare_boards(state, &scratch);
    if divergences.is_empty() {
        return;
    }
    let report = divergences
        .iter()
        .map(|d| {
            format!(
                "  {:?} {}: incremental={} full={}",
                d.object,
                d.class.label(),
                d.incremental,
                d.full
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    panic!(
        "CR 613.1 differential-flush divergence: the entry-incremental pass \
         produced a board different from a full evaluation of the same \
         post-entry state ({} field(s)):\n{report}",
        divergences.len()
    );
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::game::game_object::{DisplaySource, GameObject};
    use crate::game::layers::{flush_layers, reset_recipient_to_base};
    use crate::game::zones::create_object;
    use crate::types::ability::{
        AbilityDefinition, AbilityKind, ContinuousModification, Effect, ReplacementDefinition,
        StaticDefinition, TargetFilter, TypeFilter, TypedFilter,
    };
    use crate::types::card::{PrintedCardRef, TokenImageRef};
    use crate::types::card_type::CoreType;
    use crate::types::identifiers::CardId;
    use crate::types::keywords::Keyword;
    use crate::types::mana::{ManaColor, ManaCost, ManaCostShard};
    use crate::types::player::PlayerId;
    use crate::types::replacements::ReplacementEvent;
    use crate::types::statics::StaticMode;

    /// Two green bears under a plain "+1/+1 to creatures you control" anthem —
    /// a board whose derived P/T is produced by a continuous effect, so an
    /// incremental pass that forgot to apply the anthem to a new entry (or that
    /// clobbered the pre-existing bears) is visible in the comparison.
    fn anthem_board() -> GameState {
        let mut state = GameState::new_two_player(42);
        for i in 0..2 {
            let id = create_object(
                &mut state,
                CardId(400 + i),
                PlayerId(0),
                format!("Bear{i}"),
                Zone::Battlefield,
            );
            let ts = state.next_timestamp();
            let o = state.objects.get_mut(&id).unwrap();
            o.card_types.core_types.push(CoreType::Creature);
            o.base_card_types = o.card_types.clone();
            o.power = Some(2);
            o.toughness = Some(2);
            o.base_power = Some(2);
            o.base_toughness = Some(2);
            o.timestamp = ts;
        }
        let anthem = create_object(
            &mut state,
            CardId(410),
            PlayerId(0),
            "Plain Anthem".to_string(),
            Zone::Battlefield,
        );
        let ts = state.next_timestamp();
        let o = state.objects.get_mut(&anthem).unwrap();
        o.card_types.core_types.push(CoreType::Enchantment);
        o.base_card_types = o.card_types.clone();
        o.timestamp = ts;
        o.static_definitions.push(
            StaticDefinition::new(StaticMode::Continuous)
                .affected(TargetFilter::Typed(TypedFilter::new(TypeFilter::Creature)))
                .modifications(vec![
                    ContinuousModification::AddPower { value: 1 },
                    ContinuousModification::AddToughness { value: 1 },
                ]),
        );
        state.layers_dirty = crate::types::game_state::LayersDirty::Full;
        state
    }

    /// A plain 1/1 entering the battlefield, marked the way the real ETB
    /// pipeline marks it (`mark_entered`) — `create_object` is bare scaffolding
    /// and does no dirty bookkeeping of its own.
    fn add_plain_bear(state: &mut GameState, card_id: u64) -> ObjectId {
        let id = create_object(
            state,
            CardId(card_id),
            PlayerId(0),
            "Entering Bear".to_string(),
            Zone::Battlefield,
        );
        let ts = state.next_timestamp();
        let o = state.objects.get_mut(&id).unwrap();
        o.card_types.core_types.push(CoreType::Creature);
        o.base_card_types = o.card_types.clone();
        o.power = Some(1);
        o.toughness = Some(1);
        o.base_power = Some(1);
        o.base_toughness = Some(1);
        o.timestamp = ts;
        state.layers_dirty.mark_entered(id);
        id
    }

    /// A pre-existing anthem recipient — deterministic across runs, unlike
    /// "whatever `objects` iterates first" (`im::HashMap` is unordered).
    fn bear_id(state: &GameState) -> ObjectId {
        *state
            .objects
            .iter()
            .filter(|(_, o)| o.name.starts_with("Bear"))
            .map(|(id, _)| id)
            .min()
            .expect("anthem_board puts two bears on the battlefield")
    }

    fn bear_snapshot(state: &GameState) -> RecipientDerived {
        RecipientDerived::of(state.objects.get(&bear_id(state)).unwrap())
    }

    /// Perturb exactly one compared field. Compile-time tripwire #3: the match
    /// is exhaustive over `DivergenceClass`, so a new class must be given a
    /// perturbation here before the per-class test can compile.
    fn perturb(s: &RecipientDerived, class: DivergenceClass) -> RecipientDerived {
        let mut s = s.clone();
        match class {
            // Presence is not a field of the snapshot; it is tested separately
            // by removing an object from one board.
            DivergenceClass::Presence => {}
            DivergenceClass::Name => s.name.push_str("-perturbed"),
            DivergenceClass::Power => s.power = Some(s.power.unwrap_or(0) + 1),
            DivergenceClass::Toughness => s.toughness = Some(s.toughness.unwrap_or(0) + 1),
            DivergenceClass::Loyalty => s.loyalty = Some(s.loyalty.unwrap_or(0) + 1),
            DivergenceClass::CardTypes => s.card_types.push('X'),
            DivergenceClass::ManaCost => s.mana_cost.push('X'),
            DivergenceClass::Keywords => s.keywords.push('X'),
            DivergenceClass::Abilities => s.abilities.push('X'),
            DivergenceClass::ReplacementDefinitions => s.replacement_definitions.push('X'),
            DivergenceClass::StaticDefinitions => s.static_definitions.push('X'),
            DivergenceClass::Color => s.color.push('X'),
            DivergenceClass::PrintedRef => s.printed_ref.push('X'),
            DivergenceClass::DisplaySource => s.display_source.push('X'),
            DivergenceClass::TokenImageRef => s.token_image_ref.push('X'),
            DivergenceClass::Controller => s.controller.push('X'),
            DivergenceClass::AssignsDamageFromToughness => {
                s.assigns_damage_from_toughness = !s.assigns_damage_from_toughness
            }
            DivergenceClass::AssignsDamageAsThoughUnblocked => {
                s.assigns_damage_as_though_unblocked = !s.assigns_damage_as_though_unblocked
            }
            DivergenceClass::AssignsNoCombatDamage => {
                s.assigns_no_combat_damage = !s.assigns_no_combat_damage
            }
        }
        s
    }

    #[test]
    fn identical_boards_report_no_divergence() {
        let mut state = anthem_board();
        flush_layers(&mut state);
        assert!(compare_boards(&state, &state.clone()).is_empty());
    }

    #[test]
    fn every_field_class_is_reported_under_its_own_name() {
        let mut state = anthem_board();
        flush_layers(&mut state);
        let base = bear_snapshot(&state);
        for class in ALL_CLASSES {
            if class == DivergenceClass::Presence {
                continue;
            }
            let found = compare_snapshots(ObjectId(1), &base, &perturb(&base, class));
            let classes: Vec<DivergenceClass> = found.iter().map(|d| d.class).collect();
            assert_eq!(
                classes,
                vec![class],
                "perturbing {} must report exactly that class",
                class.label()
            );
        }
    }

    #[test]
    fn every_class_is_actually_compared_not_merely_clean() {
        // A comparator that never looks reports no divergences, exactly like a
        // correct board does. The per-class counters are what tell the two
        // apart, so every class must be > 0 after a real comparison.
        let mut state = anthem_board();
        flush_layers(&mut state);
        reset_comparison_counts();
        compare_boards(&state, &state.clone());
        let counts = comparison_counts();
        for class in ALL_CLASSES {
            assert!(
                counts[class.index()] > 0,
                "class {} was never compared",
                class.label()
            );
        }
    }

    #[test]
    fn a_missing_object_is_a_presence_divergence() {
        let mut state = anthem_board();
        flush_layers(&mut state);
        let mut short = state.clone();
        let victim = bear_id(&state);
        short.objects.remove(&victim);
        let found = compare_boards(&short, &state);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].class, DivergenceClass::Presence);
        assert_eq!(found[0].object, victim);
    }

    /// One live-field perturbation, named for the reset field it stands for.
    /// The list below must cover every field `reset_recipient_to_base` writes
    /// (`seed_live_characteristics_from_base` plus the controller and the three
    /// CR 613.11 combat-assignment flags).
    type Perturbation = (&'static str, fn(&mut GameObject));

    const RESET_FIELD_PERTURBATIONS: &[Perturbation] = &[
        ("name", |o| o.name.push_str("-perturbed")),
        ("power", |o| o.power = Some(o.power.unwrap_or(0) + 7)),
        ("toughness", |o| {
            o.toughness = Some(o.toughness.unwrap_or(0) + 7)
        }),
        ("loyalty", |o| o.loyalty = Some(o.loyalty.unwrap_or(0) + 7)),
        ("card_types", |o| {
            o.card_types.core_types.push(CoreType::Artifact)
        }),
        ("mana_cost", |o| {
            o.mana_cost = ManaCost::Cost {
                shards: vec![ManaCostShard::Green],
                generic: 3,
            }
        }),
        ("keywords", |o| o.keywords.push(Keyword::Flying)),
        ("abilities", |o| {
            Arc::make_mut(&mut o.abilities)
                .push(AbilityDefinition::new(AbilityKind::Activated, Effect::NoOp))
        }),
        ("replacement_definitions", |o| {
            o.replacement_definitions =
                vec![ReplacementDefinition::new(ReplacementEvent::Draw)].into()
        }),
        ("static_definitions", |o| {
            o.static_definitions = vec![StaticDefinition::continuous()].into()
        }),
        ("color", |o| o.color.push(ManaColor::Blue)),
        ("printed_ref", |o| {
            o.printed_ref = Some(PrintedCardRef {
                oracle_id: "perturbed".to_string(),
                face_name: "perturbed".to_string(),
            })
        }),
        ("display_source", |o| {
            o.display_source = DisplaySource::Token
        }),
        ("token_image_ref", |o| {
            o.token_image_ref = Some(TokenImageRef {
                scryfall_id: "perturbed".to_string(),
                scryfall_oracle_id: None,
                face_name: None,
                preset_id: "perturbed".to_string(),
            })
        }),
        ("controller", |o| o.controller = PlayerId(1)),
        ("assigns_damage_from_toughness", |o| {
            o.assigns_damage_from_toughness = true
        }),
        ("assigns_damage_as_though_unblocked", |o| {
            o.assigns_damage_as_though_unblocked = true
        }),
        ("assigns_no_combat_damage", |o| {
            o.assigns_no_combat_damage = true
        }),
    ];

    #[test]
    fn every_reset_field_is_observed_by_the_comparator() {
        // The runtime half of the reset <-> comparator tie. For each field
        // `reset_recipient_to_base` writes, two things must hold:
        //
        //  1. perturbing the LIVE value moves the snapshot -- otherwise the
        //     comparator does not read that field, and a divergence in it would
        //     be invisible to the harness;
        //  2. the reset erases the perturbation -- reset(perturbed) equals
        //     reset(clean). Note this is NOT "reset restores the pre-reset
        //     snapshot": the board is already fully derived, so a recipient's
        //     power is 3 (2 base + the anthem) before the reset and 2 after.
        let mut state = anthem_board();
        flush_layers(&mut state);
        let id = bear_id(&state);

        let mut clean = state.clone();
        reset_recipient_to_base(clean.objects.get_mut(&id).unwrap());
        let reset_clean = RecipientDerived::of(clean.objects.get(&id).unwrap());

        for (field, perturb_obj) in RESET_FIELD_PERTURBATIONS {
            let mut scratch = state.clone();
            let obj = scratch.objects.get_mut(&id).unwrap();
            let before = RecipientDerived::of(obj);
            perturb_obj(obj);
            assert_ne!(
                before,
                RecipientDerived::of(obj),
                "reset field `{field}` is invisible to the comparator's snapshot"
            );
            reset_recipient_to_base(obj);
            assert_eq!(
                RecipientDerived::of(obj),
                reset_clean,
                "reset did not erase the perturbation of `{field}`"
            );
        }
    }

    #[test]
    fn an_incremental_entry_matches_a_full_pass() {
        // End-to-end wiring: with verification switched on for this thread, a
        // plain creature entering under a plain anthem must take the
        // incremental arm AND agree with a full re-evaluation of the same
        // post-entry board. `flush_layers` panics on divergence.
        let mut state = anthem_board();
        flush_layers(&mut state);
        add_plain_bear(&mut state, 420);
        assert!(
            matches!(
                state.layers_dirty,
                crate::types::game_state::LayersDirty::EnteredObjects(_)
            ),
            "entry must mark the incremental dirty state, not Full"
        );

        let was = set_enabled(true);
        reset_comparison_counts();
        flush_layers(&mut state);
        set_enabled(was);

        assert!(
            comparison_counts()[DivergenceClass::Power.index()] > 0,
            "verification never ran -- the harness would report clean on any board"
        );
        // The entered bear is 1/1 base and must have taken the anthem's +1/+1.
        let entered = state
            .objects
            .iter()
            .find(|(_, o)| o.name == "Entering Bear")
            .map(|(_, o)| (o.power, o.toughness))
            .unwrap();
        assert_eq!(entered, (Some(2), Some(2)));
    }

    #[test]
    #[should_panic(expected = "differential-flush divergence")]
    fn a_corrupted_incremental_board_is_caught() {
        // Proves the harness FAILS when it should: corrupt the incremental
        // board after the pass, then verify against the honest scratch.
        let mut state = anthem_board();
        flush_layers(&mut state);
        let scratch = state.clone();
        let id = bear_id(&state);
        state.objects.get_mut(&id).unwrap().power = Some(99);
        verify(&state, Some(scratch));
    }

    #[test]
    fn the_shadow_pass_is_invisible_to_the_work_counters() {
        // The harness must not change what the counter tests measure. Without
        // suppression the shadow `evaluate_layers` inflates the gather count of
        // the very flush those tests pin (observed: 3 where 1 is asserted).
        use crate::game::layers::{
            active_effect_collection_count, reset_active_effect_collection_count,
        };

        let mut state = anthem_board();
        flush_layers(&mut state);
        add_plain_bear(&mut state, 421);

        let was = set_enabled(true);
        reset_comparison_counts();
        reset_active_effect_collection_count();
        flush_layers(&mut state);
        let with_harness = active_effect_collection_count();
        set_enabled(was);

        assert!(
            comparison_counts()[DivergenceClass::Power.index()] > 0,
            "the shadow pass must actually have run for this to mean anything"
        );
        assert_eq!(
            with_harness, 1,
            "the incremental flush gathers once; the shadow pass must not be counted"
        );
    }

    #[test]
    fn disabled_verification_clones_nothing() {
        let mut state = anthem_board();
        flush_layers(&mut state);
        let was = set_enabled(false);
        let scratch = scratch_before_flush(&state);
        set_enabled(was);
        assert!(scratch.is_none());
    }
}
