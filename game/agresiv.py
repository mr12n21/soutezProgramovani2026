import sys
import json
import time

BOT_NAME = "Marek Broz AGGRO V2"
BOARD_SIZE = 61

COORDS = [None] * BOARD_SIZE
COORDS[49] = [0, -8]
COORDS[50] = [-1, -7]
COORDS[48] = [1, -7]
COORDS[51] = [-2, -6]
COORDS[28] = [0, -6]
COORDS[47] = [2, -6]
COORDS[52] = [-3, -5]
COORDS[29] = [-1, -5]
COORDS[27] = [1, -5]
COORDS[46] = [3, -5]
COORDS[53] = [-4, -4]
COORDS[30] = [-2, -4]
COORDS[13] = [0, -4]
COORDS[26] = [2, -4]
COORDS[45] = [4, -4]
COORDS[31] = [-3, -3]
COORDS[14] = [-1, -3]
COORDS[12] = [1, -3]
COORDS[25] = [3, -3]
COORDS[54] = [-4, -2]
COORDS[15] = [-2, -2]
COORDS[4] = [0, -2]
COORDS[11] = [2, -2]
COORDS[44] = [4, -2]
COORDS[32] = [-3, -1]
COORDS[5] = [-1, -1]
COORDS[3] = [1, -1]
COORDS[24] = [3, -1]
COORDS[55] = [-4, 0]
COORDS[16] = [-2, 0]
COORDS[0] = [0, 0]
COORDS[10] = [2, 0]
COORDS[43] = [4, 0]
COORDS[33] = [-3, 1]
COORDS[6] = [-1, 1]
COORDS[2] = [1, 1]
COORDS[23] = [3, 1]
COORDS[56] = [-4, 2]
COORDS[17] = [-2, 2]
COORDS[1] = [0, 2]
COORDS[9] = [2, 2]
COORDS[42] = [4, 2]
COORDS[34] = [-3, 3]
COORDS[8] = [1, 3]
COORDS[7] = [0, 4]
COORDS[22] = [3, 3]
COORDS[57] = [-4, 4]
COORDS[35] = [-2, 4]
COORDS[18] = [-1, 3]
COORDS[21] = [2, 4]
COORDS[41] = [4, 4]
COORDS[20] = [1, 5]
COORDS[40] = [3, 5]
COORDS[19] = [0, 6]
COORDS[58] = [-3, 5]
COORDS[39] = [2, 6]
COORDS[60] = [-1, 7]
COORDS[36] = [-1, 5]
COORDS[38] = [1, 7]
COORDS[59] = [-2, 6]
COORDS[37] = [0, 8]


def validate_coords(print_report=False):
    seen = set()
    duplicates = []
    missing = []

    for i, coord in enumerate(COORDS):
        if coord is None:
            missing.append(i)
            continue
        key = (coord[0], coord[1])
        if key in seen:
            duplicates.append((i, coord))
        seen.add(key)

    if print_report:
        for i, coord in duplicates:
            print("Duplicate coordinate:", i, coord)
        print("unique coords:", len(seen))
        if missing:
            print("missing indices:", json.dumps(missing))

    return duplicates, missing


BLOCKED = {0, 39, 43, 47, 51, 55, 59}
P1_START = {41, 49, 57}
P2_START = {37, 45, 53}


def get_distance(i1, i2):
    x1, y1 = COORDS[i1]
    x2, y2 = COORDS[i2]
    # This coordinate map uses unit hex steps; this metric matches appendix scanline mapping.
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, (dx + dy) // 2)


DIST_MATRIX = [[get_distance(i, j) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
NEIGHBORS_1 = [[j for j in range(BOARD_SIZE) if j != i and DIST_MATRIX[i][j] == 1] for i in range(BOARD_SIZE)]
NEIGHBORS_2 = [[j for j in range(BOARD_SIZE) if j != i and DIST_MATRIX[i][j] == 2] for i in range(BOARD_SIZE)]

INF = 10**12
MAX_SEARCH_DEPTH = 9
DEFAULT_TIME_BUDGET_SECONDS = 0.55
ENDGAME_SOLVE_FREE_CELLS = 8

# Heuristic weights inspired by strong Hexxagon engines.
W_STONES = 0.7
W_CONVERTIBLE = 3.4
W_POSITION = 2.0
W_BLOCK = 1.2
W_CONNECTIVITY = 0.2
W_PRESSURE = 2.8
W_SAFETY = 0.6
W_ATTACK_MAP = 1.35
W_FRONTLINE = 1.4
W_INVASION = 1.1


def _positional_cell_weight(index):
    if index in BLOCKED:
        return 0
    ring = DIST_MATRIX[0][index]
    if ring <= 1:
        return 1
    if ring >= 4:
        return 1
    return 0


POSITIONAL_CELL_WEIGHT = [_positional_cell_weight(i) for i in range(BOARD_SIZE)]
TRANSPOSITION_TABLE = {}
TRANSPOSITION_MAX_SIZE = 200000
HISTORY_TABLE = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def make_initial_state():
    state = [0] * BOARD_SIZE
    for idx in BLOCKED:
        state[idx] = -1
    for idx in P1_START:
        state[idx] = 1
    for idx in P2_START:
        state[idx] = 2
    return state


def other_player(player):
    return 2 if player == 1 else 1


def has_any_move(state, player):
    for start in range(BOARD_SIZE):
        if state[start] != player:
            continue
        for end in NEIGHBORS_1[start]:
            if state[end] == 0:
                return True
        for end in NEIGHBORS_2[start]:
            if state[end] == 0:
                return True
    return False


def generate_moves(state, player):
    moves = []
    for start in range(BOARD_SIZE):
        if state[start] != player:
            continue
        for end in NEIGHBORS_1[start]:
            if state[end] == 0:
                moves.append((start, end))
        for end in NEIGHBORS_2[start]:
            if state[end] == 0:
                moves.append((start, end))
    return moves


def apply_move(state, start, end):
    if not (0 <= start < BOARD_SIZE and 0 <= end < BOARD_SIZE):
        return False

    player = state[start]
    if player not in (1, 2):
        return False
    if state[end] != 0:
        return False

    d = DIST_MATRIX[start][end]
    if d not in (1, 2):
        return False

    if d == 2:
        state[start] = 0
    state[end] = player

    opponent = other_player(player)
    for nb in NEIGHBORS_1[end]:
        if state[nb] == opponent:
            state[nb] = player

    # Special end rule from appendix/task 4.
    if not has_any_move(state, opponent):
        for i in range(BOARD_SIZE):
            if state[i] == 0:
                state[i] = player

    return True


def game_over_result(state):
    # Task 6: game continues while at least one free cell exists.
    if 0 in state:
        return "false"

    p1 = state.count(1)
    p2 = state.count(2)
    if p1 > p2:
        return "true player1"
    if p2 > p1:
        return "true player2"
    return "true draw"


def evaluate_state(state, player):
    opponent = other_player(player)

    stones = state.count(player) - state.count(opponent)

    # Opponent stones adjacent to our stones are tactical conversion targets.
    own_convertible = 0
    opp_convertible = 0
    own_positional = 0
    opp_positional = 0
    for idx, value in enumerate(state):
        if value == player:
            own_positional += POSITIONAL_CELL_WEIGHT[idx]
            own_convertible += sum(1 for nb in NEIGHBORS_1[idx] if state[nb] == opponent)
        elif value == opponent:
            opp_positional += POSITIONAL_CELL_WEIGHT[idx]
            opp_convertible += sum(1 for nb in NEIGHBORS_1[idx] if state[nb] == player)

    own_moves_list = generate_moves(state, player)
    opp_moves_list = generate_moves(state, opponent)
    own_moves = len(own_moves_list)
    opp_moves = len(opp_moves_list)

    own_pressure = 0
    opp_pressure = 0
    own_risk = 0
    opp_risk = 0
    for idx, value in enumerate(state):
        if value not in (player, opponent):
            continue

        allied = 0
        enemy = 0
        for nb in NEIGHBORS_1[idx]:
            if state[nb] == value:
                allied += 1
            elif state[nb] == (opponent if value == player else player):
                enemy += 1

        if value == player:
            own_pressure += enemy
            own_risk += max(0, enemy - allied)
        else:
            opp_pressure += enemy
            opp_risk += max(0, enemy - allied)

    own_attack_map = 0
    for _, end in own_moves_list:
        own_attack_map += sum(1 for nb in NEIGHBORS_1[end] if state[nb] == opponent)

    opp_attack_map = 0
    for _, end in opp_moves_list:
        opp_attack_map += sum(1 for nb in NEIGHBORS_1[end] if state[nb] == player)

    own_frontline = 0
    opp_frontline = 0
    own_invasion = 0
    opp_invasion = 0
    for idx, value in enumerate(state):
        if value not in (player, opponent):
            continue

        enemy_adjacent = sum(
            1 for nb in NEIGHBORS_1[idx] if state[nb] == (opponent if value == player else player)
        )
        if enemy_adjacent == 0:
            continue

        if value == player:
            own_frontline += 1
            own_invasion += enemy_adjacent
        else:
            opp_frontline += 1
            opp_invasion += enemy_adjacent

    block_score = 0
    if opp_moves == 0:
        block_score += 1
    if own_moves == 0:
        block_score -= 1

    own_group = largest_group_size(state, player)
    opp_group = largest_group_size(state, opponent)

    value = 0.0
    value += W_STONES * stones
    aggression_response = 1.0 + max(0.0, (opp_pressure - own_pressure) * 0.04)
    value += W_CONVERTIBLE * aggression_response * (own_convertible - opp_convertible)
    value += W_POSITION * (own_positional - opp_positional)
    value += W_BLOCK * block_score
    value += W_CONNECTIVITY * (own_group - opp_group)
    value += W_PRESSURE * (own_pressure - opp_pressure)
    value += W_SAFETY * (opp_risk - own_risk)
    value += W_ATTACK_MAP * (own_attack_map - opp_attack_map)
    value += W_FRONTLINE * (own_frontline - opp_frontline)
    value += W_INVASION * (own_invasion - opp_invasion)

    # Small mobility nudge to avoid self-traps in midgame.
    value += 0.25 * (own_moves - opp_moves)

    return int(value * 100)


def terminal_score(state, player):
    opponent = other_player(player)
    return (state.count(player) - state.count(opponent)) * 100000


def largest_group_size(state, player):
    visited = set()
    best = 0

    for idx, value in enumerate(state):
        if value != player or idx in visited:
            continue

        stack = [idx]
        visited.add(idx)
        size = 0

        while stack:
            node = stack.pop()
            size += 1
            for nb in NEIGHBORS_1[node]:
                if state[nb] == player and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if size > best:
            best = size

    return best


def tactical_move_score(state, player, move):
    start, end = move
    opponent = other_player(player)

    clone_bonus = 1 if DIST_MATRIX[start][end] == 1 else 0
    jump_bonus = 2 if DIST_MATRIX[start][end] == 2 else 0
    converted = sum(1 for nb in NEIGHBORS_1[end] if state[nb] == opponent)
    positional = POSITIONAL_CELL_WEIGHT[end]

    end_support = sum(1 for nb in NEIGHBORS_1[end] if state[nb] == player)
    end_danger = sum(1 for nb in NEIGHBORS_1[end] if state[nb] == opponent)
    start_support = sum(1 for nb in NEIGHBORS_1[start] if state[nb] == player)

    frontier_bonus = end_danger * 2

    jump_penalty = 0
    if DIST_MATRIX[start][end] == 2 and start_support == 0 and converted == 0:
        # Even this aggro profile avoids pointless long jumps with no pressure gain.
        jump_penalty = 1

    return (
        converted * 8
        + clone_bonus
        + jump_bonus
        + positional
        + end_support
        + frontier_bonus
        - jump_penalty
    )


def _history_score(move):
    start, end = move
    return HISTORY_TABLE[start][end]


def _killer_bonus(move, killer_moves, ply):
    killers = killer_moves.get(ply, ())
    if killers and move == killers[0]:
        return 8000
    if len(killers) > 1 and move == killers[1]:
        return 5000
    return 0


def order_moves(state, player, moves, pv_move=None, killer_moves=None, ply=0):
    def move_order_key(move):
        score = tactical_move_score(state, player, move) * 100
        score += _history_score(move)
        if killer_moves is not None:
            score += _killer_bonus(move, killer_moves, ply)
        return score

    ordered = sorted(moves, key=move_order_key, reverse=True)
    if pv_move is not None and pv_move in ordered:
        ordered.remove(pv_move)
        ordered.insert(0, pv_move)
    return ordered


def target_depth_for_state(state):
    free_cells = state.count(0)
    if free_cells <= ENDGAME_SOLVE_FREE_CELLS:
        return 12
    if free_cells <= 10:
        return MAX_SEARCH_DEPTH
    if free_cells <= 18:
        return 8
    if free_cells <= 30:
        return 7
    return 6


def _register_killer(killer_moves, ply, move):
    entries = list(killer_moves.get(ply, ()))
    if move in entries:
        entries.remove(move)
    entries.insert(0, move)
    killer_moves[ply] = tuple(entries[:2])


def compute_time_budget_seconds(state):

    return 0.86


def negamax(state, current_player, depth, alpha, beta, deadline, killer_moves, ply):
    if time.monotonic() >= deadline:
        raise TimeoutError()

    alpha_original = alpha
    key = (tuple(state), current_player)
    cached = TRANSPOSITION_TABLE.get(key)
    pv_move = None
    if cached is not None and cached["depth"] >= depth:
        cached_value = cached["value"]
        cached_flag = cached["flag"]
        pv_move = cached.get("best_move")
        if cached_flag == "exact":
            return cached_value
        if cached_flag == "lower":
            alpha = max(alpha, cached_value)
        elif cached_flag == "upper":
            beta = min(beta, cached_value)
        if alpha >= beta:
            return cached_value

    if 0 not in state:
        return terminal_score(state, current_player)
    if depth == 0:
        return evaluate_state(state, current_player)

    moves = generate_moves(state, current_player)
    if not moves:
        if not has_any_move(state, other_player(current_player)):
            return terminal_score(state, current_player)
        return -negamax(state, other_player(current_player), depth - 1, -beta, -alpha, deadline, killer_moves, ply + 1)

    best_value = -INF
    best_move = None

    ordered_moves = order_moves(
        state,
        current_player,
        moves,
        pv_move=pv_move,
        killer_moves=killer_moves,
        ply=ply,
    )

    for idx, (start, end) in enumerate(ordered_moves):
        child = state[:]
        apply_move(child, start, end)

        if idx == 0:
            value = -negamax(
                child,
                other_player(current_player),
                depth - 1,
                -beta,
                -alpha,
                deadline,
                killer_moves,
                ply + 1,
            )
        else:
            value = -negamax(
                child,
                other_player(current_player),
                depth - 1,
                -alpha - 1,
                -alpha,
                deadline,
                killer_moves,
                ply + 1,
            )
            if alpha < value < beta:
                value = -negamax(
                    child,
                    other_player(current_player),
                    depth - 1,
                    -beta,
                    -alpha,
                    deadline,
                    killer_moves,
                    ply + 1,
                )

        if value > best_value:
            best_value = value
            best_move = (start, end)

        alpha = max(alpha, best_value)
        if alpha >= beta:
            _register_killer(killer_moves, ply, (start, end))
            HISTORY_TABLE[start][end] += depth * depth
            break

    flag = "exact"
    if best_value <= alpha_original:
        flag = "upper"
    elif best_value >= beta:
        flag = "lower"

    TRANSPOSITION_TABLE[key] = {
        "depth": depth,
        "value": best_value,
        "flag": flag,
        "best_move": best_move,
    }

    return best_value


def choose_next_move(state, player):
    moves = generate_moves(state, player)
    if not moves:
        return []

    if len(TRANSPOSITION_TABLE) > TRANSPOSITION_MAX_SIZE:
        TRANSPOSITION_TABLE.clear()

    opponent = other_player(player)
    killer_moves = {}
    ordered = order_moves(state, player, moves)
    best_move = ordered[0]
    completed_best = best_move

    depth_limit = target_depth_for_state(state)
    deadline = time.monotonic() + compute_time_budget_seconds(state)

    aspiration_width = 600
    previous_score = None

    for depth in range(1, depth_limit + 1):
        if time.monotonic() >= deadline:
            break

        iteration_best_value = -INF
        iteration_best_move = completed_best
        if previous_score is None:
            alpha = -INF
            beta = INF
        else:
            alpha = previous_score - aspiration_width
            beta = previous_score + aspiration_width
        timed_out = False

        while True:
            current_alpha = alpha
            failed = False

            for start, end in ordered:
                child = state[:]
                apply_move(child, start, end)

                try:
                    value = -negamax(
                        child,
                        opponent,
                        depth - 1,
                        -beta,
                        -current_alpha,
                        deadline,
                        killer_moves,
                        1,
                    )
                except TimeoutError:
                    timed_out = True
                    failed = True
                    break

                if value > iteration_best_value:
                    iteration_best_value = value
                    iteration_best_move = (start, end)

                current_alpha = max(current_alpha, iteration_best_value)

            if timed_out:
                break

            if iteration_best_value <= alpha:
                alpha -= aspiration_width
                beta = (previous_score + aspiration_width) if previous_score is not None else INF
                aspiration_width *= 2
                iteration_best_value = -INF
                iteration_best_move = completed_best
                failed = True
            elif iteration_best_value >= beta:
                alpha = (previous_score - aspiration_width) if previous_score is not None else -INF
                beta += aspiration_width
                aspiration_width *= 2
                iteration_best_value = -INF
                iteration_best_move = completed_best
                failed = True

            if not failed:
                break

        if timed_out:
            break

        completed_best = iteration_best_move
        previous_score = iteration_best_value
        aspiration_width = max(300, aspiration_width // 2)
        ordered = order_moves(
            state,
            player,
            ordered,
            pv_move=completed_best,
            killer_moves=killer_moves,
            ply=0,
        )

    return [completed_best[0], completed_best[1]]


def choose_and_apply_next_move(state, player):
    move = choose_next_move(state, player)
    if not move:
        return []

    start, end = move
    if apply_move(state, start, end):
        return [start, end]

    for alt_start, alt_end in generate_moves(state, player):
        if apply_move(state, alt_start, alt_end):
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
        if 0 <= idx < BOARD_SIZE:
            write_json(COORDS[idx])
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
        if 0 <= i1 < BOARD_SIZE and 0 <= i2 < BOARD_SIZE:
            write_line(str(DIST_MATRIX[i1][i2]))
        else:
            write_line("false")
        return False

    if line.startswith("play_move"):
        payload = line[len("play_move"):].strip()
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
            write_line("true" if apply_move(state, move[0], move[1]) else "false")
        else:
            write_line("false")
        return False

    if line == "board_state":
        write_json(state)
        return False

    if line == "game_over":
        write_line(game_over_result(state))
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
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-coords":
        validate_coords(print_report=True)
        return

    state = make_initial_state()

    # Optional single-command mode from CLI arguments.
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
