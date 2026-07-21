//! Subprocess-level regression tests for `ai-commander`'s `--games-file`
//! batch mode, specifically the per-line feed override (pod-lab
//! simulation-acceleration plan, Tier 1 follow-up: each `--games-file` line
//! may now resolve its own feed, not just its own seed/difficulty). A batch
//! invocation's per-game output must match single-game invocations of the
//! same seed+feed, byte-for-byte outside the one inherently nondeterministic
//! field (wall-clock elapsed time). These spawn the real `ai-commander`
//! binary rather than calling `run()` in-process, since the behavior under
//! test — argument parsing, per-line feed routing, stdout flush timing, and
//! process exit — is the binary's actual contract with the pod-lab harness,
//! not an internal function's.
//!
//! `#[ignore]` because every test here loads `client/public/card-data.json`
//! (requires `cargo run --bin card-data-export` or the setup.sh script),
//! which is not available in unit-test CI — same convention as
//! `greasefang_bounded.rs`/`whitemane_lion_bounded.rs`. Opt in via
//! `cargo test -p phase-ai --test ai_commander_batch_equivalence -- --ignored`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Resolves `client/public` the same way `greasefang_bounded.rs` et al. do:
/// `PHASE_CARDS_PATH` override, else relative to this crate's manifest dir.
fn cards_dir() -> PathBuf {
    std::env::var("PHASE_CARDS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("client")
                .join("public")
        })
}

/// Small action cap so these tests run quickly while still exercising
/// several turns. The exact outcome (COMPLETED/ABORT/STALL) doesn't matter
/// for equivalence — only that a given seed+feed+cap combination reaches the
/// SAME outcome deterministically, single-game or batched.
const TEST_ACTION_CAP: &str = "300";

/// Writes a second commander-style feed, distinct from the process default
/// (`feeds/mtggoldfish-commander.json`), by reversing that feed's `decks`
/// array — the first 4 decks `run`/`resolve_feed_payload` resolve are then a
/// different set of commanders/decklists in different seats, while every
/// card name is guaranteed to already exist in this build's card-data.json
/// (they're copied verbatim from the real feed, just reordered). This
/// proves per-line feed ROUTING, not just that the game engine is
/// deterministic: if a batch line silently reused the wrong feed, the
/// resulting game would very likely diverge from the correct single-game
/// run given a fully different 4-deck table.
fn write_reversed_feed(dest: &Path) {
    let source = cards_dir().join("feeds").join("mtggoldfish-commander.json");
    let mut feed: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&source).unwrap_or_else(|e| panic!("open {}: {e}", source.display())),
    )
    .expect("mtggoldfish-commander.json is valid JSON");
    feed["decks"]
        .as_array_mut()
        .expect("feed.decks is an array")
        .reverse();
    std::fs::write(dest, serde_json::to_string(&feed).unwrap())
        .unwrap_or_else(|e| panic!("write {}: {e}", dest.display()));
}

/// Runs the real `ai-commander` binary with `cards_dir()` as its positional
/// arg and `args` appended, and returns captured stdout as a `String`. Exit
/// code isn't asserted here: COMPLETED (0)/ABORT (2)/STALL (3) are all
/// legitimate outcomes for a bounded-action-cap test game, and every one of
/// them still prints a full `=== RESULT ===` block — `normalized_result_block`
/// panics if that block is missing, which already catches an actual crash.
fn run_ai_commander(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_ai-commander"))
        .arg(cards_dir())
        .args(args)
        .output()
        .expect("spawn ai-commander");
    String::from_utf8(output.stdout).expect("stdout is valid UTF-8")
}

/// Splits `stdout` into one chunk per game. Batch mode anchors on the
/// `--- GAME ` marker: each chunk then runs from one game's own marker up to
/// (not including) the next game's marker or EOF, so it's fully
/// self-contained — critically, a game's marker + tier echo + feed-resolve
/// diagnostics (all printed BEFORE that game's own "Game started." line)
/// stay attributed to THAT game, not leaked onto the end of the previous
/// one. Single-game mode never prints a marker, so the whole output is one
/// chunk.
fn game_blocks(stdout: &str) -> Vec<&str> {
    const MARKER: &str = "--- GAME ";
    if !stdout.contains(MARKER) {
        return vec![stdout];
    }
    let mut starts: Vec<usize> = stdout.match_indices(MARKER).map(|(i, _)| i).collect();
    starts.push(stdout.len());
    starts.windows(2).map(|w| &stdout[w[0]..w[1]]).collect()
}

/// The `=== RESULT ===` epilogue through the end of one game's block (as
/// isolated by `game_blocks`), with the single inherently nondeterministic
/// line (`Elapsed: {:.1}s`, wall-clock) stripped so two separately-timed runs
/// can be compared for equality.
fn normalized_result_block(game_block: &str) -> String {
    let start = game_block
        .find("=== RESULT ===")
        .expect("game block contains a RESULT epilogue");
    game_block[start..]
        .lines()
        .filter(|line| !line.starts_with("Elapsed:"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
#[ignore = "loads card-data.json + runs real games; opt in via --ignored"]
fn batch_per_line_feed_matches_single_game_across_two_feeds() {
    let tmp = std::env::temp_dir();
    let pid = std::process::id();
    let feed_b_path = tmp.join(format!("ai_commander_batch_eq_feed_b_{pid}.json"));
    write_reversed_feed(&feed_b_path);
    let feed_b_arg = feed_b_path.to_str().expect("temp path is valid UTF-8");

    let seed_a = "9001";
    let seed_b = "9002";
    let feed_a_arg = "feeds/mtggoldfish-commander.json";

    // Single-game baselines, one per feed.
    let single_a = run_ai_commander(&[
        "--seed",
        seed_a,
        "--difficulty",
        "Easy",
        "--feed",
        feed_a_arg,
        "--action-cap",
        TEST_ACTION_CAP,
    ]);
    let single_b = run_ai_commander(&[
        "--seed",
        seed_b,
        "--difficulty",
        "Easy",
        "--feed",
        feed_b_arg,
        "--action-cap",
        TEST_ACTION_CAP,
    ]);

    // One batch invocation, per-line feed override on each line — game 1
    // names feed A explicitly, game 2 names feed B, proving a batch can mix
    // distinct feeds across lines via the LRU-cached resolver.
    let games_file_path = tmp.join(format!("ai_commander_batch_eq_games_{pid}.txt"));
    std::fs::write(
        &games_file_path,
        format!("{seed_a},Easy,{feed_a_arg}\n{seed_b},Easy,{feed_b_arg}\n"),
    )
    .expect("write games-file");
    let batch = run_ai_commander(&[
        "--games-file",
        games_file_path.to_str().unwrap(),
        "--action-cap",
        TEST_ACTION_CAP,
    ]);

    let _ = std::fs::remove_file(&feed_b_path);
    let _ = std::fs::remove_file(&games_file_path);

    let single_a_blocks = game_blocks(&single_a);
    let single_b_blocks = game_blocks(&single_b);
    let batch_blocks = game_blocks(&batch);

    assert_eq!(
        single_a_blocks.len(),
        1,
        "single-game A must print exactly one game"
    );
    assert_eq!(
        single_b_blocks.len(),
        1,
        "single-game B must print exactly one game"
    );
    assert_eq!(
        batch_blocks.len(),
        2,
        "batch must print exactly one block per games-file line"
    );

    // Asserts the test's own premise before trusting the equivalence checks
    // below: feed A and feed B must actually produce DIFFERENT games (a
    // different 4-deck table plus a different seed), so a batch run that
    // silently routed every line to the wrong feed — or ignored the 3rd
    // field entirely — would show up as a mismatch here, not slip through
    // as coincidental equality. Without this, the two `assert_eq!`s below
    // could pass even if per-line feed routing were completely broken.
    let single_a_result = normalized_result_block(single_a_blocks[0]);
    let single_b_result = normalized_result_block(single_b_blocks[0]);
    assert_ne!(
        single_a_result, single_b_result,
        "feed A and feed B must produce distinguishable single-game RESULT \
         blocks, or the equivalence checks below can't discriminate correct \
         per-line feed routing from a broken batch that used the same feed \
         for every line — adjust the seeds/feed to force a difference"
    );

    assert_eq!(
        single_a_result,
        normalized_result_block(batch_blocks[0]),
        "batch game 1 (feed A via 3rd-field override) must match single-game A"
    );
    assert_eq!(
        single_b_result,
        normalized_result_block(batch_blocks[1]),
        "batch game 2 (feed B via 3rd-field override) must match single-game B"
    );
}

#[test]
#[ignore = "loads card-data.json + runs real games; opt in via --ignored"]
fn batch_output_echoes_seed_and_parsed_difficulty_per_game() {
    let pid = std::process::id();
    let games_file_path = std::env::temp_dir().join(format!("ai_commander_echo_games_{pid}.txt"));
    // Both lines omit the 3rd field (fall back to the process default feed)
    // — this test is only about the per-game tier echo, not feed routing.
    std::fs::write(&games_file_path, "9101,Easy\n9102,VeryHard\n").expect("write games-file");

    let batch = run_ai_commander(&[
        "--games-file",
        games_file_path.to_str().unwrap(),
        "--action-cap",
        TEST_ACTION_CAP,
    ]);
    let _ = std::fs::remove_file(&games_file_path);

    assert!(
        batch.contains("--- GAME seed=9101 difficulty=Easy ---\nSeed: 9101   Difficulty: Easy\n"),
        "game 1 must echo its parsed seed+difficulty immediately after its marker:\n{batch}"
    );
    assert!(
        batch.contains(
            "--- GAME seed=9102 difficulty=VeryHard ---\nSeed: 9102   Difficulty: VeryHard\n"
        ),
        "game 2 must echo its parsed seed+difficulty immediately after its marker:\n{batch}"
    );
}

#[test]
#[ignore = "loads card-data.json + runs real games; opt in via --ignored"]
fn single_game_stdout_is_deterministic_and_preamble_is_pinned() {
    let seed = "9201";
    let out1 = run_ai_commander(&[
        "--seed",
        seed,
        "--difficulty",
        "Easy",
        "--action-cap",
        TEST_ACTION_CAP,
    ]);
    let out2 = run_ai_commander(&[
        "--seed",
        seed,
        "--difficulty",
        "Easy",
        "--action-cap",
        TEST_ACTION_CAP,
    ]);

    // Two runs of the same seed/feed/difficulty must produce identical
    // output modulo wall-clock timing — proving single-game mode's output is
    // still fully determined by its inputs after the per-line-feed refactor
    // (`resolve_feed_payload`, the exact function single-game mode also
    // calls, exactly once, for the process-level default feed).
    let normalize = |s: &str| {
        s.lines()
            .filter(|l| !l.contains("elapsed=") && !l.starts_with("Elapsed:"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(normalize(&out1), normalize(&out2));

    // Pins the exact preamble single-game mode has always printed, in
    // order — the lines the per-line-feed refactor's extraction of
    // `resolve_feed_payload` touched most directly. A reordering (e.g.
    // "Feed:" moving relative to "Seed:.../Difficulty:...") would silently
    // break the pod-lab harness's stdout parsing; this fails loudly instead.
    let expected_preamble = format!(
        "=== 4-player Commander AI test ===\n\
         Feed: feeds/mtggoldfish-commander.json\n\
         Seed: {seed}   Difficulty: Easy\n\n"
    );
    assert!(
        out1.starts_with(&expected_preamble),
        "single-game preamble format changed:\n{out1}"
    );
}
