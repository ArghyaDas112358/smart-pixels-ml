"""
One seed -> one colour, for every O11 figure in the deck.

Having each plotting script keep its own dict is how seed 1042 ended up teal in
one figure and navy in the next. Import from here instead.

The same hexes are used by the slide chrome in
atrain-slides/scripts/build_weekly_deck.mjs (C.blue / C.green / C.amber).
"""

SEEDS = [4042, 1042, 2042, 3042]

ACCENT = {
    4042: "#1d4ed8",   # royal blue
    1042: "#1e3a8a",   # deep navy
    2042: "#15803d",   # green   -- the standout seed
    3042: "#b45309",   # amber
}

# Gautschi O11 seeds, for when the deck grows to all eight.
ACCENT_EXTRA = {
    6042: "#7c3aed",   # violet
    5042: "#0e7490",   # cyan-teal
    9042: "#be123c",   # rose
    7042: "#4d7c0f",   # olive
}

# O4 (beta-annealing) seeds -- deliberately a different family from the O11 four,
# so an O4 panel is never mistaken for an O11 one at a glance.
SEEDS_O4 = [8042, 8142, 8242, 8342]
ACCENT_O4 = {
    8042: "#0f766e",   # teal
    8142: "#6d28d9",   # violet
    8242: "#be123c",   # rose
    8342: "#4d7c0f",   # olive
}


# O13 (two-lambda conditional-NLL) seeds.
SEEDS_O13 = [13042, 13142, 13342, 13442]
ACCENT_O13 = {
    13142: "#15803d",   # green  -- best cot alpha
    13442: "#1d4ed8",   # blue
    13042: "#b45309",   # amber
    13342: "#6d28d9",   # violet
}


# O21a.v2-cold (leakage-free pair-lattice, loss v2) seeds. Violet leads because
# it is the O21a mechanism colour on the slides; the winner seed carries it.
SEEDS_COLD = [22042, 22142, 22242]
ACCENT_COLD = {
    22042: "#7c3aed",   # violet -- found the wide pair from cold
    22142: "#0e7490",   # cyan-teal -- mid-window basin
    22242: "#b45309",   # amber -- mid-window basin
}


# O21a.v2-phi0 (AF A100 cold-from-0 seeds, COMPLETE phi record from epoch 0).
# 23042 carries violet because it is the one that found the basin.
SEEDS_PHI0 = [23042, 23142, 23242, 23342]
ACCENT_PHI0 = {
    23042: "#7c3aed",   # violet -- found the basin, (11,21)
    23142: "#b45309",   # amber/orange -- stuck (30,31); matches its slide theme
    23242: "#be123c",   # rose -- stuck (53,56)
    23342: "#7f1d1d",   # deep red -- stuck (65,73); matches its slide theme
}


def color(seed, default="#334155"):
    """Colour for a seed, falling back to slate for anything unregistered."""
    return ACCENT.get(seed, ACCENT_O13.get(seed, ACCENT_O4.get(
        seed, ACCENT_COLD.get(seed, ACCENT_PHI0.get(seed, ACCENT_EXTRA.get(seed, default))))))
