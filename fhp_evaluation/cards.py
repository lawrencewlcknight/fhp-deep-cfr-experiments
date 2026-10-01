"""Card conversion, state parsing, and an independently implemented hand ranker."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable, Sequence


RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANKS)}
CARD_RE = re.compile(r"[2-9TJQKA][cdhs]")


def card_name(card_id: int) -> str:
    if not 0 <= int(card_id) < 52:
        raise ValueError(f"Card id outside [0, 51]: {card_id}")
    return RANKS[int(card_id) // 4] + SUITS[int(card_id) % 4]


def card_id(card: str) -> int:
    if len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS:
        raise ValueError(f"Invalid card: {card!r}")
    return 4 * RANKS.index(card[0]) + SUITS.index(card[1])


def parse_cards(value: str) -> tuple[str, ...]:
    cards = tuple(CARD_RE.findall(value))
    if "".join(cards) != value:
        raise ValueError(f"Could not parse card string: {value!r}")
    if len(set(cards)) != len(cards):
        raise ValueError("A card appears more than once")
    return cards


def cards_from_information_state(state, player: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    text = state.information_state_string(int(player))
    private_match = re.search(r"\[Private: ([^\]]*)\]", text)
    public_match = re.search(r"\[Public: ([^\]]*)\]", text)
    if private_match is None or public_match is None:
        raise ValueError(f"Unsupported OpenSpiel information-state string: {text!r}")
    private = tuple(card_id(card) for card in parse_cards(private_match.group(1)))
    public = tuple(card_id(card) for card in parse_cards(public_match.group(1)))
    return private, public


def canonical_starting_hand(cards: Sequence[int]) -> str:
    if len(cards) != 2 or cards[0] == cards[1]:
        raise ValueError("A starting hand must contain two distinct cards")
    names = [card_name(card) for card in cards]
    names.sort(key=lambda value: RANK_VALUE[value[0]], reverse=True)
    high, low = names
    if high[0] == low[0]:
        return high[0] + low[0]
    suffix = "s" if high[1] == low[1] else "o"
    return high[0] + low[0] + suffix


def five_card_rank(cards: Iterable[int]) -> tuple[int, ...]:
    """Return a totally ordered standard-poker rank for exactly five cards."""
    cards = tuple(int(card) for card in cards)
    if len(cards) != 5 or len(set(cards)) != 5:
        raise ValueError("Exactly five distinct cards are required")
    names = [card_name(card) for card in cards]
    ranks = [RANK_VALUE[name[0]] for name in names]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({name[1] for name in names}) == 1
    unique = sorted(set(ranks))
    if unique == [2, 3, 4, 5, 14]:
        straight_high = 5
    elif len(unique) == 5 and unique[-1] - unique[0] == 4:
        straight_high = unique[-1]
    else:
        straight_high = 0
    if flush and straight_high:
        return (8, straight_high)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight_high:
        return (4, straight_high)
    if groups[0][0] == 3:
        kickers = sorted((rank for rank in ranks if rank != groups[0][1]), reverse=True)
        return (3, groups[0][1], *kickers)
    pairs = sorted((rank for rank, count in counts.items() if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = next(rank for rank, count in counts.items() if count == 1)
        return (2, *pairs, kicker)
    if len(pairs) == 1:
        kickers = sorted((rank for rank in ranks if rank != pairs[0]), reverse=True)
        return (1, pairs[0], *kickers)
    return (0, *sorted(ranks, reverse=True))


def showdown_result(hero: Sequence[int], opponent: Sequence[int], board: Sequence[int]) -> float:
    """Return 1 for a hero win, 1/2 for a tie, and 0 for a loss in FHP."""
    if len(hero) != 2 or len(opponent) != 2 or len(board) != 3:
        raise ValueError("FHP showdown requires two hole cards per player and three board cards")
    all_cards = tuple(hero) + tuple(opponent) + tuple(board)
    if len(set(all_cards)) != 7:
        raise ValueError("Showdown contains duplicate cards")
    hero_rank = five_card_rank(tuple(hero) + tuple(board))
    opponent_rank = five_card_rank(tuple(opponent) + tuple(board))
    return 1.0 if hero_rank > opponent_rank else 0.5 if hero_rank == opponent_rank else 0.0

