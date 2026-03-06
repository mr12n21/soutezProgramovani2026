import json
import sys
import time

import main as core

BOT_NAME = "Marek Broz BASIC"
INF = 10**12
TT_MAX_SIZE = 120000

# Basic varianta ma jednodussi evaluaci, ale hleda hluboko.
OPENING_DEPTH = 7
MIDGAME_DEPTH = 9
LATEGAME_DEPTH = 11
ENDGAME_DEPTH = 12

OPENING_TIME = 0.70
MIDGAME_TIME = 0.85
LATEGAME_TIME = 1.05
ENDGAME_TIME = 1.20

TRANSPOSITION_TABLE = {}


def other_player(player):
    return 2 if player == 1 else 1


def connected_components_stats(state, player):
    visited = set()
    largest = 0
    groups = 0

    for idx, value in enumerate(state):
        if value != player or idx in visited:
            continue

        groups += 1
        stack = [idx]
        visited.add(idx)
        size = 0

        while stack:
            node = stack.pop()
            size += 1
            for nb in core.NEIGHBORS_1[node]:
                if state[nb] == player and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if size > largest:
            largest = size

    return largest, groups


def evaluate_state(state, player):
    opponent = other_player(player)

    my_stones = state.count(player)
    opp_stones = state.count(opponent)
    stones_diff = my_stones - opp_stones

    my_largest, my_groups = connected_components_stats(state, player)
    opp_largest, opp_groups = connected_components_stats(state, opponent)

    my_moves = len(core.generate_moves(state, player))
    opp_moves = len(core.generate_moves(state, opponent))

    # Tlak na rust shluku:
    # - vetsi nejvetsi skupina je dobra
    # - mene skupin je dobra (kompaktnost)
    cluster_size_term = my_largest - opp_largest
    compactness_term = opp_groups - my_groups
    mobility_term = my_moves - opp_moves

    score = 0
    score += 120 * stones_diff
    score += 220 * cluster_size_term
    score += 70 * compactness_term
    score += 25 * mobility_term

    return score


def terminal_score(state, player):
    opponent = other_player(player)
    return (state.count(player) - state.count(opponent)) * 100000


def tactical_move_score(state, player, move):
    start, end = move
    opponent = other_player(player)

    converted = sum(1 for nb in core.NEIGHBORS_1[end] if state[nb] == opponent)
    support = sum(1 for nb in core.NEIGHBORS_1[end] if state[nb] == player)
    clone_bonus = 2 if core.DIST_MATRIX[start][end] == 1 else 0
    jump_penalty = 2 if (core.DIST_MATRIX[start][end] == 2 and support == 0) else 0

    return converted * 8 + support * 2 + clone_bonus - jump_penalty


def order_moves(state, player, moves, preferred=None):
    ordered = sorted(moves, key=lambda mv: tactical_move_score(state, player, mv), reverse=True)
    if preferred is not None and preferred in ordered:
        ordered.remove(preferred)
        ordered.insert(0, preferred)
    return ordered


def target_depth_for_state(state):
    free_cells = state.count(0)
    if free_cells <= 8:
        return ENDGAME_DEPTH
    if free_cells <= 18:
        return LATEGAME_DEPTH
    if free_cells <= 34:
        return MIDGAME_DEPTH
    return OPENING_DEPTH


def time_budget_seconds(state):
    free_cells = state.count(0)
    if free_cells <= 8:
        return ENDGAME_TIME
    if free_cells <= 18:
        return LATEGAME_TIME
    if free_cells <= 34:
        return MIDGAME_TIME
    return OPENING_TIME


def negamax(state, current_player, depth, alpha, beta, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError()

    key = (tuple(state), current_player, depth)
    cached = TRANSPOSITION_TABLE.get(key)
    if cached is not None:
        return cached

    if 0 not in state:
        return terminal_score(state, current_player)
    if depth == 0:
        return evaluate_state(state, current_player)

    moves = core.generate_moves(state, current_player)
    if not moves:
        opponent = other_player(current_player)
        if not core.has_any_move(state, opponent):
            return terminal_score(state, current_player)
        return -negamax(state, opponent, depth - 1, -beta, -alpha, deadline)

    best = -INF
    ordered = order_moves(state, current_player, moves)

    for start, end in ordered:
        child = state[:]
        core.apply_move(child, start, end)
        val = -negamax(child, other_player(current_player), depth - 1, -beta, -alpha, deadline)

        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if len(TRANSPOSITION_TABLE) > TT_MAX_SIZE:
        TRANSPOSITION_TABLE.clear()
    TRANSPOSITION_TABLE[key] = best
    return best


def choose_next_move(state, player):
    moves = core.generate_moves(state, player)
    if not moves:
        return []

    deadline = time.monotonic() + time_budget_seconds(state)
    depth_limit = target_depth_for_state(state)

    ordered_root = order_moves(state, player, moves)
    best_move = ordered_root[0]
    best_value = -INF

    for depth in range(1, depth_limit + 1):
        if time.monotonic() >= deadline:
            break

        current_best_move = best_move
        current_best_value = -INF
        timed_out = False

        ordered_root = order_moves(state, player, ordered_root, preferred=best_move)

        for start, end in ordered_root:
            child = state[:]
            core.apply_move(child, start, end)

            try:
                value = -negamax(child, other_player(player), depth - 1, -INF, INF, deadline)
            except TimeoutError:
                timed_out = True
                break

            if value > current_best_value:
                current_best_value = value
                current_best_move = (start, end)

        if timed_out:
            break

        best_move = current_best_move
        best_value = current_best_value

    _ = best_value
    return [best_move[0], best_move[1]]


def choose_and_apply_next_move(state, player):
    move = choose_next_move(state, player)
    if not move:
        return []

    start, end = move
    if core.apply_move(state, start, end):
        return [start, end]

    for alt_start, alt_end in core.generate_moves(state, player):
        if core.apply_move(state, alt_start, alt_end):
            return [alt_start, alt_end]
    return []


def parse_player(token):
    t = token.strip().lower()
    if t in ("player1", "1", "p1"):
        return 1
    if t in ("player2", "2", "p2"):
        return 2
    return None


def write_line(text):
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def write_json(value):
    write_line(json.dumps(value, separators=(",", ":")))


def process_command(state, line):
    line = line.strip()
    if not line:
        return False

    if line == "get_name":
        write_line(BOT_NAME)
        return False

    if line == "exit":
        return True

    if line.startswith("index_to_position"):
        parts = line.split()
        if len(parts) != 2:
            write_line("false")
            return False
        try:
            idx = int(parts[1])
        except ValueError:
            write_line("false")
            return False
        if 0 <= idx < core.BOARD_SIZE:
            write_json(core.COORDS[idx])
        else:
            write_line("false")
        return False

    if line.startswith("distance_between"):
        parts = line.split()
        if len(parts) != 3:
            write_line("false")
            return False
        try:
            i1 = int(parts[1])
            i2 = int(parts[2])
        except ValueError:
            write_line("false")
            return False
        if 0 <= i1 < core.BOARD_SIZE and 0 <= i2 < core.BOARD_SIZE:
            write_line(str(core.DIST_MATRIX[i1][i2]))
        else:
            write_line("false")
        return False

    if line.startswith("play_move"):
        payload = line[len("play_move") :].strip()
        try:
            move = json.loads(payload)
        except Exception:
            write_line("false")
            return False
        if (
            isinstance(move, list)
            and len(move) == 2
            and isinstance(move[0], int)
            and isinstance(move[1], int)
        ):
            write_line("true" if core.apply_move(state, move[0], move[1]) else "false")
        else:
            write_line("false")
        return False

    if line == "board_state":
        write_json(state)
        return False

    if line == "game_over":
        write_line(core.game_over_result(state))
        return False

    if line.startswith("next_move"):
        parts = line.split()
        if len(parts) != 2:
            write_json([])
            return False
        player = parse_player(parts[1])
        if player is None:
            write_json([])
            return False
        write_json(choose_and_apply_next_move(state, player))
        return False

    write_line("false")
    return False


def main():
    state = core.make_initial_state()

    if len(sys.argv) > 1:
        line = " ".join(sys.argv[1:])
        process_command(state, line)
        return

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        should_exit = process_command(state, line)
        if should_exit:
            break


if __name__ == "__main__":
    main()
