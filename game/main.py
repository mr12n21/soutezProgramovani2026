import sys
import json

BOT_NAME = "BOT_HEX_2026"
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
    mobility = len(generate_moves(state, player)) - len(generate_moves(state, opponent))
    return stones * 100 + mobility * 6


def negamax(state, current_player, maximizing_player, depth, alpha, beta):
    if depth == 0 or 0 not in state:
        return evaluate_state(state, maximizing_player)

    moves = generate_moves(state, current_player)
    if not moves:
        if not has_any_move(state, other_player(current_player)):
            return evaluate_state(state, maximizing_player)
        return -negamax(state, other_player(current_player), maximizing_player, depth - 1, -beta, -alpha)

    best = -10**9
    for start, end in moves:
        child = state[:]
        apply_move(child, start, end)
        score = -negamax(child, other_player(current_player), maximizing_player, depth - 1, -beta, -alpha)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def choose_next_move(state, player):
    moves = generate_moves(state, player)
    if not moves:
        return []

    # Move ordering by immediate tactical gain.
    opponent = other_player(player)
    scored = []
    for start, end in moves:
        gain = 1 if DIST_MATRIX[start][end] == 1 else 0
        converted = 0
        for nb in NEIGHBORS_1[end]:
            if state[nb] == opponent:
                converted += 1
        scored.append((gain + converted, (start, end)))
    scored.sort(key=lambda item: item[0], reverse=True)
    ordered = [m for _, m in scored]

    best_move = ordered[0]
    best_value = -10**9

    for start, end in ordered:
        child = state[:]
        apply_move(child, start, end)
        value = -negamax(child, opponent, player, 1, -10**9, 10**9)
        if value > best_value:
            best_value = value
            best_move = (start, end)

    for start, end in ordered[:8]:
        child = state[:]
        apply_move(child, start, end)
        value = -negamax(child, opponent, player, 2, -10**9, 10**9)
        if value > best_value:
            best_value = value
            best_move = (start, end)

    return [best_move[0], best_move[1]]


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
