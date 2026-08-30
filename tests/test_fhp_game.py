from deep_cfr_poker.game import FHP_GAME_PARAMETERS, load_fhp_game


def test_fhp_parameters_match_existing_repository_contract():
    assert dict(FHP_GAME_PARAMETERS) == {
        "betting": "limit",
        "blind": "50 100",
        "raiseSize": "100 100",
        "firstPlayer": "1 2",
        "maxRaises": "3 3",
        "numRounds": 2,
        "numSuits": 4,
        "numRanks": 13,
        "numHoleCards": 2,
        "numBoardCards": "0 3",
        "numPlayers": 2,
    }


def test_open_spiel_loads_expected_fhp_shape():
    game = load_fhp_game()
    assert game.num_players() == 2
    assert game.num_distinct_actions() == 3
    assert game.information_state_tensor_size() == 190
    assert game.max_game_length() == 28
    assert game.max_utility() == 700.0
    assert game.min_utility() == -700.0

    state = game.new_initial_state()
    chance_actions = 0
    while state.is_chance_node():
        action = state.chance_outcomes()[0][0]
        state.apply_action(action)
        chance_actions += 1
    assert chance_actions == 4
    assert state.current_player() == 0
    assert state.legal_actions() == [0, 1, 2]
