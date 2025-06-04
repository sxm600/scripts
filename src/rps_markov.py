import sys

from collections import defaultdict


POSSIBLE_MOVES = ['rock', 'paper', 'scissors']

WINNING_MOVES = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock'
}


class HiddenMarkovModel:
    def __init__(self, transition_matrix: defaultdict[str, defaultdict[str, float]] = None) -> None:
        if transition_matrix is None:
            self.transition_matrix = defaultdict(lambda: defaultdict(float))
        else:
            self.transition_matrix = transition_matrix

    def predict_next_state(self, current_state: str) -> str | None:
        next_states = self.transition_matrix[current_state]

        if next_states:
            return max(next_states, key=next_states.get)

        return None

    def learn_transition(self, state_from: str, state_to: str, learning_rate: float = 0.5) -> None:
        self.transition_matrix[state_from][state_to] += learning_rate

        weights_sum = sum(list(self.transition_matrix[state_from].values()))

        # normalize so sum of weights equals to 1.0
        for target in self.transition_matrix[state_from].keys():
            self.transition_matrix[state_from][target] /= weights_sum


# creates fully connected graph with equal weights for rps
def create_rps_transition_matrix() -> defaultdict[str, defaultdict[str, float]]:
    transition_matrix = defaultdict(lambda: defaultdict(float))

    for state_from in POSSIBLE_MOVES:
        for state_to in POSSIBLE_MOVES:
            transition_matrix[state_from][state_to] = 1 / 3

    return transition_matrix


def ask_point_limit() -> int:
    while not (user_input := input('To how many points shall we play? > ')).isdigit():
        print('Only numbers allowed! e.g. 3, 10, 20...')

    return int(user_input)


def main() -> None:
    rps_transition_matrix = create_rps_transition_matrix()
    hmm = HiddenMarkovModel(rps_transition_matrix)

    print('Welcome to "Rock, Paper, Scissors" game against Markov\'s Model !!!')
    print('You play by typing one of the valid moves (rock, paper, scissors) or (quit) to stop the game.')

    point_limit = ask_point_limit()
    last_move = 'rock'
    player_points = 0
    ai_points = 0

    while player_points < point_limit and ai_points < point_limit:
        player_move = input('Your move > ').strip().lower()

        if player_move == 'quit':
            print('See you next time!')
            sys.exit()

        if player_move not in POSSIBLE_MOVES:
            print(f'There is no such move as "{player_move}", possible moves are (rock, paper, scissors)')
            continue

        prediction = hmm.predict_next_state(last_move)
        ai_move = WINNING_MOVES[prediction]

        print(f'Moves: AI - {ai_move}, Player - {player_move}')

        if player_move == ai_move:
            print('Draw!')
        elif WINNING_MOVES[player_move] == ai_move:
            print('Computer has won! 0_0')
            ai_points += 1
        else:
            print('Player has won!')
            player_points += 1

        print(f'Score: AI - {ai_points}, Player - {player_points}')

        hmm.learn_transition(last_move, player_move)
        last_move = player_move


    print(f'Final Score: AI - {ai_points}, Player - {player_points}!')


if __name__ == '__main__':
    main()