# It's the survival of the biggest!
# Propel your chips across a frictionless table top to avoid getting eaten by bigger foes.
# Aim for smaller oil droplets for an easy size boost.
# Tip: merging your chips will give you a sizeable advantage.

player_id = int(input())  # your id (0 to 4)

# game loop
while True:
    player_chip_count = int(input())  # The number of chips under your control
    entity_count = int(
        input()
    )  # The total number of entities on the table, including your chips
    for i in range(entity_count):
        inputs = input().split()
        _id = int(inputs[0])  # Unique identifier for this entity
        player = int(inputs[1])  # The owner of this entity (-1 for neutral droplets)
        radius = float(inputs[2])  # the radius of this entity
        x = float(inputs[3])  # the X coordinate (0 to 799)
        y = float(inputs[4])  # the Y coordinate (0 to 514)
        vx = float(inputs[5])  # the speed of this entity along the X axis
        vy = float(inputs[6])  # the speed of this entity along the Y axis
    for i in range(player_chip_count):

        # Write an action using print
        # To debug: print("Debug messages...", file=sys.stderr, flush=True)

        # One instruction per chip: 2 real numbers (x y) for a propulsion, or 'WAIT'.
        print("0 0")
