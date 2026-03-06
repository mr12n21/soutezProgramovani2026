import sys
import json


BOT_NAME = "BOT_HEX_2026"

COORDS = [
	[0, 0], [0, 2], [2, 0], [1, -1], [0, -2], [-1, -1], [-2, 0], [0, 4], [1, 3], [2, 2],
	[3, 1], [2, -2], [1, -3], [0, -4], [-1, -3], [-2, -2], [-2, 0], [-2, 2], [-1, 3], [0, 6],
	[1, 5], [2, 4], [3, 3], [4, 2], [4, 0], [3, -1], [2, -4], [1, -5], [0, -6], [-1, -5],
	[-2, -4], [-3, -1], [-4, 0], [-4, 2], [-3, 3], [-2, 4], [-1, 5], [0, 8], [1, 7], [2, 6],
	[3, 5], [4, 4], [4, 2], [4, 0], [4, -2], [4, -4], [3, -5], [2, -6], [1, -7], [0, -8],
	[-1, -7], [-2, -6], [-3, -5], [-4, -4], [-4, -2], [-4, 0], [-4, 2], [-4, 4], [-3, 5], [-2, 6],
	[-1, 7],
]

BOARD_SIZE = 61

BLOCKED = {0, 39, 43, 47, 51, 55, 59}
P1_START = {41, 49, 57}
P2_START = {37, 45, 53}


def make_initial_state():
	state = [0] * BOARD_SIZE
	for idx in BLOCKED:
		state[idx] = -1
	for idx in P1_START:
		state[idx] = 1
	for idx in P2_START:
		state[idx] = 2
	return state


def distance(i1, i2):
	x1, y1 = COORDS[i1]
	x2, y2 = COORDS[i2]
	return max(abs(x1 - x2), abs(y1 - y2), abs((x1 + y1) - (x2 + y2))) // 2


DIST = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
NEIGHBORS_1 = [[] for _ in range(BOARD_SIZE)]
NEIGHBORS_2 = [[] for _ in range(BOARD_SIZE)]

for i in range(BOARD_SIZE):
	for j in range(BOARD_SIZE):
		d = distance(i, j)
		DIST[i][j] = d
		if i != j and d == 1:
			NEIGHBORS_1[i].append(j)
		elif i != j and d == 2:
			NEIGHBORS_2[i].append(j)


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
	# Depth 2 for all moves, then selective depth 3 on top candidates.
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
	player = state[start]
	if player not in (1, 2):
		return False
	if not (0 <= end < BOARD_SIZE):
		return False
	if state[end] != 0:
		return False

	d = DIST[start][end]
	if d not in (1, 2):
		return False

	if d == 2:
		state[start] = 0
	state[end] = player

	opponent = other_player(player)
	for nb in NEIGHBORS_1[end]:
		if state[nb] == opponent:
			state[nb] = player

	if not has_any_move(state, opponent):
		for i in range(BOARD_SIZE):
			if state[i] == 0:
				state[i] = player

	return True


def game_over_result(state):
	if 0 in state:
		return "false"

	p1 = 0
	p2 = 0
	for value in state:
		if value == 1:
			p1 += 1
		elif value == 2:
			p2 += 1

	if p1 > p2:
		return "true player1"
	if p2 > p1:
		return "true player2"
	return "true draw"


def evaluate_state(state, player):
	opponent = other_player(player)
	my_count = 0
	opp_count = 0
	my_frontier = 0
	opp_frontier = 0

	for i, val in enumerate(state):
		if val == player:
			my_count += 1
			for nb in NEIGHBORS_1[i]:
				if state[nb] == 0:
					my_frontier += 1
					break
		elif val == opponent:
			opp_count += 1
			for nb in NEIGHBORS_1[i]:
				if state[nb] == 0:
					opp_frontier += 1
					break

	mobility = len(generate_moves(state, player)) - len(generate_moves(state, opponent))
	return (my_count - opp_count) * 100 + mobility * 4 + (my_frontier - opp_frontier)


def simulate_move(state, move):
	new_state = state[:]
	ok = apply_move(new_state, move[0], move[1])
	return new_state if ok else None


def ordered_moves(state, player):
	moves = generate_moves(state, player)
	scored = []
	opponent = other_player(player)
	for move in moves:
		end = move[1]
		d = DIST[move[0]][end]
		gain = 1 if d == 1 else 0
		converted = 0
		for nb in NEIGHBORS_1[end]:
			if state[nb] == opponent:
				converted += 1
		scored.append((gain + converted, move))
	scored.sort(key=lambda x: x[0], reverse=True)
	return [m for _, m in scored]


def negamax(state, current, maximizing_for, depth, alpha, beta):
	if depth == 0 or 0 not in state:
		return evaluate_state(state, maximizing_for)

	moves = ordered_moves(state, current)
	if not moves:
		if not has_any_move(state, other_player(current)):
			return evaluate_state(state, maximizing_for)
		return -negamax(state, other_player(current), maximizing_for, depth - 1, -beta, -alpha)

	best = -10**18
	for move in moves:
		child = simulate_move(state, move)
		if child is None:
			continue
		score = -negamax(child, other_player(current), maximizing_for, depth - 1, -beta, -alpha)
		if score > best:
			best = score
		if best > alpha:
			alpha = best
		if alpha >= beta:
			break
	return best


def choose_next_move(state, player):
	moves = ordered_moves(state, player)
	if not moves:
		return []

	best_move = moves[0]
	best_score = -10**18

	scored_depth2 = []
	for move in moves:
		child = simulate_move(state, move)
		if child is None:
			continue
		score = -negamax(child, other_player(player), player, 1, -10**18, 10**18)
		scored_depth2.append((score, move))
		if score > best_score:
			best_score = score
			best_move = move

	scored_depth2.sort(key=lambda x: x[0], reverse=True)
	top_for_deeper = scored_depth2[:6]

	for _, move in top_for_deeper:
		child = simulate_move(state, move)
		if child is None:
			continue
		score = -negamax(child, other_player(player), player, 2, -10**18, 10**18)
		if score > best_score:
			best_score = score
			best_move = move

	return [best_move[0], best_move[1]]


def parse_player_token(token):
	t = token.strip().lower()
	if t in ("player1", "1", "p1"):
		return 1
	if t in ("player2", "2", "p2"):
		return 2
	return None


def print_line(value):
	sys.stdout.write(value + "\n")
	sys.stdout.flush()


def print_json(value):
	print_line(json.dumps(value, separators=(",", ":")))


def main():
	state = make_initial_state()

	while True:
		line = sys.stdin.readline()
		if not line:
			break
		line = line.strip()
		if not line:
			continue

		if line == "get_name":
			print_line(BOT_NAME)
			continue

		if line == "exit":
			break

		if line.startswith("index_to_position"):
			parts = line.split()
			if len(parts) != 2:
				print_line("false")
				continue
			try:
				idx = int(parts[1])
				if 0 <= idx < BOARD_SIZE:
					print_json(COORDS[idx])
				else:
					print_line("false")
			except ValueError:
				print_line("false")
			continue

		if line.startswith("distance_between"):
			parts = line.split()
			if len(parts) != 3:
				print_line("false")
				continue
			try:
				i1 = int(parts[1])
				i2 = int(parts[2])
				if 0 <= i1 < BOARD_SIZE and 0 <= i2 < BOARD_SIZE:
					print_line(str(DIST[i1][i2]))
				else:
					print_line("false")
			except ValueError:
				print_line("false")
			continue

		if line.startswith("play_move"):
			payload = line[len("play_move"):].strip()
			try:
				move = json.loads(payload)
				if (
					isinstance(move, list)
					and len(move) == 2
					and isinstance(move[0], int)
					and isinstance(move[1], int)
					and 0 <= move[0] < BOARD_SIZE
					and 0 <= move[1] < BOARD_SIZE
				):
					ok = apply_move(state, move[0], move[1])
					print_line("true" if ok else "false")
				else:
					print_line("false")
			except Exception:
				print_line("false")
			continue

		if line == "board_state":
			print_json(state)
			continue

		if line == "game_over":
			print_line(game_over_result(state))
			continue

		if line.startswith("next_move"):
			parts = line.split()
			if len(parts) != 2:
				print_json([])
				continue
			player = parse_player_token(parts[1])
			if player is None:
				print_json([])
				continue
			print_json(choose_next_move(state, player))
			continue

		print_line("false")


if __name__ == "__main__":
	main()
