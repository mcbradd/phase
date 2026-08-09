//! Telemetry coverage for full-layer-eval attribution (`FullEvalClass` /
//! `EscalationReason`).
//!
//! CR 613.1: a full layer pass is always a correct answer, so nothing asserted
//! here is a rules claim. These tests pin WHICH mutation category each
//! production marking site reports, so a `layers-attribution` readout can be
//! trusted to point at a real narrowing opportunity instead of a mislabelled
//! one.
//!
//! Every test drives a production entry point (`zones::move_to_zone`,
//! `pairing::pair_objects`, `phasing::phase_out_object`,
//! `layers::prune_end_of_turn_effects`, `layers::flush_layers`) — never the
//! marking helper directly — so reverting a site's class assignment fails the
//! assertion.

use engine::game::game_object::{BestowFormState, PhaseOutCause};
use engine::game::layers::{flush_layers, mark_layers_entered, prune_end_of_turn_effects};
use engine::game::perf_counters::{self, LayersAttribution};
use engine::game::zones::{create_object, move_to_zone};
use engine::game::{pairing, phasing};
use engine::types::ability::{ContinuousModification, Duration, TargetFilter};
use engine::types::card_type::CoreType;
use engine::types::counter::CounterType;
use engine::types::events::GameEvent;
use engine::types::game_state::{EscalationReason, FullEvalClass, GameState};
use engine::types::identifiers::{CardId, ObjectId};
use engine::types::player::PlayerId;
use engine::types::zones::Zone;

fn make_creature(state: &mut GameState, name: &str, zone: Zone) -> ObjectId {
    let id = create_object(state, CardId(0), PlayerId(0), name.to_string(), zone);
    let timestamp = state.next_timestamp();
    let object = state.objects.get_mut(&id).expect("created object exists");
    object.card_types.core_types.push(CoreType::Creature);
    object.base_card_types = object.card_types.clone();
    object.power = Some(2);
    object.toughness = Some(2);
    object.base_power = Some(2);
    object.base_toughness = Some(2);
    object.timestamp = timestamp;
    id
}

/// Drain whatever the fixture setup owed and zero the counters, so each test's
/// assertions describe exactly one window opened by the mutation under test.
/// `GameState::new_two_player` starts `Full`, and several fixture helpers mark
/// on their own, so this is required for the window counts to mean anything.
fn settle(state: &mut GameState) {
    flush_layers(state);
    perf_counters::reset();
    perf_counters::layers_attribution_reset();
    assert!(
        state.layers_full_classes.is_empty(),
        "settle() must leave no pending class marks"
    );
}

fn windows(attr: &LayersAttribution, class: FullEvalClass) -> u64 {
    attr.class_windows[class.index()]
}

/// Assert that `expected` is exactly the set of classes credited with a window.
fn assert_classes(attr: &LayersAttribution, expected: &[FullEvalClass]) {
    let actual: Vec<FullEvalClass> = FullEvalClass::ALL
        .into_iter()
        .filter(|c| windows(attr, *c) > 0)
        .collect();
    assert_eq!(
        actual, expected,
        "credited classes differ (windows: full={}, unattributed={})",
        attr.full_windows, attr.unattributed_windows
    );
}

// --------------------------------------------------------------------------
// zones.rs `move_to_zone` multi-bit arm: the only site whose class choice is
// computed rather than constant, so each axis gets its own fixture.
// --------------------------------------------------------------------------

#[test]
fn library_to_hand_move_is_classed_hand_churn_only() {
    let mut state = GameState::new_two_player(42);
    let card = make_creature(&mut state, "Drawn Card", Zone::Library);
    settle(&mut state);

    let mut events: Vec<GameEvent> = Vec::new();
    move_to_zone(&mut state, card, Zone::Hand, &mut events);
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_eq!(attr.unattributed_windows, 0);
    assert_classes(&attr, &[FullEvalClass::HandChurn]);
}

#[test]
fn battlefield_to_graveyard_move_is_classed_battlefield_exit_not_hand_churn() {
    let mut state = GameState::new_two_player(42);
    let creature = make_creature(&mut state, "Dying Creature", Zone::Battlefield);
    settle(&mut state);

    let mut events: Vec<GameEvent> = Vec::new();
    move_to_zone(&mut state, creature, Zone::Graveyard, &mut events);
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_classes(&attr, &[FullEvalClass::BattlefieldExit]);
}

#[test]
fn hand_to_battlefield_entry_is_classed_both_entry_and_hand_churn() {
    let mut state = GameState::new_two_player(42);
    let creature = make_creature(&mut state, "Played Creature", Zone::Hand);
    settle(&mut state);

    let mut events: Vec<GameEvent> = Vec::new();
    move_to_zone(&mut state, creature, Zone::Battlefield, &mut events);
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    // Overlap is intended: the same window is both a Hand/Exile-originated
    // entry and hand churn, and each axis is independently a narrowing target.
    assert_classes(
        &attr,
        &[FullEvalClass::EntryHandExile, FullEvalClass::HandChurn],
    );
}

/// Reach guard for the three tests above: a plain Graveyard -> Battlefield
/// entry takes the `mark_layers_entered` carve-out, so it opens NO full window
/// at all. Without this, the "class X is absent" halves above would also pass
/// if the marking block were skipped entirely.
#[test]
fn plain_battlefield_entry_takes_incremental_path_and_opens_no_full_window() {
    let mut state = GameState::new_two_player(42);
    let creature = make_creature(&mut state, "Reanimated Creature", Zone::Graveyard);
    settle(&mut state);

    let mut events: Vec<GameEvent> = Vec::new();
    move_to_zone(&mut state, creature, Zone::Battlefield, &mut events);
    flush_layers(&mut state);

    let counters = perf_counters::snapshot();
    assert_eq!(counters.layers_incremental, 1, "expected the fast path");
    assert_eq!(counters.layers_escalated, 0);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 0);
    assert_eq!(attr.unattributed_windows, 0);
    assert_classes(&attr, &[]);
}

/// `zones.rs`'s bestow revert is a DIRECT (`layers_dirty.mark_full()` +
/// `layers_full_classes.insert(..)`) site rather than a
/// `mark_layers_full_classed` call, and it fires inside the same window as the
/// battlefield exit — so one window legitimately carries two classes from two
/// different marking mechanisms.
#[test]
fn bestowed_aura_leaving_battlefield_is_classed_form_change_and_exit() {
    let mut state = GameState::new_two_player(42);
    let aura = make_creature(&mut state, "Bestowed Aura", Zone::Battlefield);
    state
        .objects
        .get_mut(&aura)
        .expect("aura exists")
        .bestow_form = Some(BestowFormState);
    settle(&mut state);

    let mut events: Vec<GameEvent> = Vec::new();
    move_to_zone(&mut state, aura, Zone::Graveyard, &mut events);
    flush_layers(&mut state);

    assert!(
        state.objects[&aura].bestow_form.is_none(),
        "the revert must actually have run, or FormChange would be vacuous"
    );
    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_classes(
        &attr,
        &[FullEvalClass::BattlefieldExit, FullEvalClass::FormChange],
    );
}

// --------------------------------------------------------------------------
// Converted wrapper call sites, one per representative class.
// --------------------------------------------------------------------------

#[test]
fn soulbond_pairing_is_classed_attach() {
    let mut state = GameState::new_two_player(42);
    let first = make_creature(&mut state, "Soulbond Host", Zone::Battlefield);
    let second = make_creature(&mut state, "Soulbond Partner", Zone::Battlefield);
    settle(&mut state);

    pairing::pair_objects(&mut state, first, second, PlayerId(0));
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_classes(&attr, &[FullEvalClass::Attach]);
}

#[test]
fn phasing_out_a_permanent_is_classed_phasing() {
    let mut state = GameState::new_two_player(42);
    let creature = make_creature(&mut state, "Phasing Creature", Zone::Battlefield);
    settle(&mut state);

    let mut events: Vec<GameEvent> = Vec::new();
    let phased =
        phasing::phase_out_object(&mut state, creature, PhaseOutCause::Directly, &mut events);
    assert_eq!(phased, vec![creature], "the phase-out must have happened");
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_classes(&attr, &[FullEvalClass::Phasing]);
}

#[test]
fn end_of_turn_effect_prune_is_classed_transient_effect() {
    let mut state = GameState::new_two_player(42);
    let creature = make_creature(&mut state, "Pumped Creature", Zone::Battlefield);
    state.add_transient_continuous_effect(
        creature,
        PlayerId(0),
        Duration::UntilEndOfTurn,
        TargetFilter::SpecificObject { id: creature },
        vec![ContinuousModification::AddPower { value: 1 }],
        None,
    );
    settle(&mut state);

    prune_end_of_turn_effects(&mut state);
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_classes(&attr, &[FullEvalClass::TransientEffect]);
}

// --------------------------------------------------------------------------
// Window accounting.
// --------------------------------------------------------------------------

/// Several marks between two flushes collapse into ONE window that credits
/// every class; `full_windows` must not be double-counted. Marking order must
/// not matter, so the same pair is driven in both orders.
#[test]
fn multiple_classes_in_one_window_share_a_single_full_window() {
    for reverse in [false, true] {
        let mut state = GameState::new_two_player(42);
        let first = make_creature(&mut state, "Soulbond Host", Zone::Battlefield);
        let second = make_creature(&mut state, "Soulbond Partner", Zone::Battlefield);
        state.add_transient_continuous_effect(
            first,
            PlayerId(0),
            Duration::UntilEndOfTurn,
            TargetFilter::SpecificObject { id: first },
            vec![ContinuousModification::AddPower { value: 1 }],
            None,
        );
        settle(&mut state);

        if reverse {
            prune_end_of_turn_effects(&mut state);
            pairing::pair_objects(&mut state, first, second, PlayerId(0));
        } else {
            pairing::pair_objects(&mut state, first, second, PlayerId(0));
            prune_end_of_turn_effects(&mut state);
        }
        flush_layers(&mut state);

        let attr = perf_counters::layers_attribution_snapshot();
        assert_eq!(attr.full_windows, 1, "reverse={reverse}");
        assert_classes(
            &attr,
            &[FullEvalClass::Attach, FullEvalClass::TransientEffect],
        );
    }
}

/// The conversion-coverage signal: an unconverted site still calls
/// `LayersDirty::mark_full` with no class, and that window must land in
/// `unattributed_windows` rather than being silently credited to a class.
#[test]
fn unclassed_mark_lands_in_the_unattributed_bucket() {
    let mut state = GameState::new_two_player(42);
    make_creature(&mut state, "Bystander", Zone::Battlefield);
    settle(&mut state);

    state.layers_dirty.mark_full();
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_eq!(attr.unattributed_windows, 1);
    assert_classes(&attr, &[]);
}

/// A class mark left on a window that never becomes `Full` is drained by
/// `flush_layers`, so it cannot leak into the NEXT full window's attribution.
#[test]
fn class_mark_on_a_non_full_window_does_not_leak_forward() {
    let mut state = GameState::new_two_player(42);
    make_creature(&mut state, "Bystander", Zone::Battlefield);
    settle(&mut state);

    // Stray class mark with a Clean lattice: nothing to flush, mark drained.
    state.layers_full_classes.insert(FullEvalClass::Combat);
    flush_layers(&mut state);
    assert_eq!(perf_counters::layers_attribution_snapshot().full_windows, 0);

    state.layers_dirty.mark_full();
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 1);
    assert_eq!(
        attr.unattributed_windows, 1,
        "the stray Combat mark must not have been carried into this window"
    );
    assert_classes(&attr, &[]);
}

// --------------------------------------------------------------------------
// Escalation reasons: `prepare_incremental_flush`'s `Result` arms.
// --------------------------------------------------------------------------

#[test]
fn entered_object_that_already_left_is_classed_entered_missing() {
    let mut state = GameState::new_two_player(42);
    settle(&mut state);

    mark_layers_entered(&mut state, ObjectId(9_999));
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    // An escalation is counted as an escalation, never as a classed full window.
    assert_eq!(attr.full_windows, 0);
    assert_eq!(attr.unattributed_windows, 0);
    assert_eq!(
        attr.escalation_windows[EscalationReason::EnteredMissing.index()],
        1
    );
    assert_eq!(perf_counters::snapshot().layers_escalated, 1);
}

#[test]
fn entered_object_with_counters_is_classed_entered_blocks_incremental() {
    let mut state = GameState::new_two_player(42);
    let creature = make_creature(&mut state, "Countered Creature", Zone::Battlefield);
    state
        .objects
        .get_mut(&creature)
        .expect("creature exists")
        .counters
        .insert(CounterType::Plus1Plus1, 1);
    settle(&mut state);

    mark_layers_entered(&mut state, creature);
    flush_layers(&mut state);

    let attr = perf_counters::layers_attribution_snapshot();
    assert_eq!(attr.full_windows, 0);
    assert_eq!(
        attr.escalation_windows[EscalationReason::EnteredBlocksIncremental.index()],
        1
    );
    for reason in EscalationReason::ALL {
        if reason != EscalationReason::EnteredBlocksIncremental {
            assert_eq!(
                attr.escalation_windows[reason.index()],
                0,
                "unexpected {} escalation",
                reason.name()
            );
        }
    }
}

/// Dense-index invariant the telemetry arrays depend on: `ALL` must be
/// duplicate-free and ordered by `index()`, or every per-class counter above
/// would silently address the wrong slot.
#[test]
fn class_and_reason_indices_are_dense_and_ordered() {
    for (i, class) in FullEvalClass::ALL.into_iter().enumerate() {
        assert_eq!(class.index(), i, "{} is out of order", class.name());
    }
    assert_eq!(FullEvalClass::ALL.len(), FullEvalClass::COUNT);
    for (i, reason) in EscalationReason::ALL.into_iter().enumerate() {
        assert_eq!(reason.index(), i, "{} is out of order", reason.name());
    }
    assert_eq!(EscalationReason::ALL.len(), EscalationReason::COUNT);
}
