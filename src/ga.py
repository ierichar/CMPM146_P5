import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    """ Loops through each tile possibly replacing it"""
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        #randomGen = random.seed()

        #Current: 10% chance to replace any given tile with one of the tiles below,
        #as long as the given tile is not a pipe
        tiles = ['-', 'X', '?', 'M', 'B', 'o']
        chance_replace = 0.05

        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                '''Correct pipes with no proper top'''
                #If space above genome[x][y] is valid and is not T or |
                    #if genomexy is |
                        #genomexy = T
                        #break
                '''Correct pipes with invalid blocks in the middle of them
                (That were not already corrected by generate_children())'''
                #else If space above genome[x][y] is valid and is T or |,
                    #if genomexy is not X, or |:
                        #genomexy = X
                        # (Doing X would mean pipes can be on ground that
                        # is in midair. Doing | means that unless generated
                        # that way, all pipes will go towards the ground)
                        #break
                if y > 0 and genome[y-1][x] != "T" and genome[y-1][x] != "|":
                    if genome[y][x] == "|":
                        genome[y][x] = "T"
                        break
                elif y > 0 and genome[y-1][x] == "T" or genome[y-1][x] == "|":
                    if genome[y][x] != "X" and genome[y][x] != "|":
                        genome[y][x] = "X"
                        break

                # new_roll = random.randrange(1.0)
                # if chance_replace > new_roll and genome[y][x] != 'T' and genome[y][x] != '|':
                #     genome[y][x] = random.choice(tiles)
        return genome

    # Create zero or more children from self and other
    """ Does this by looping through every tile in the level.txt, going up(?) row by row
    choose to take self's tile or other's tile for the new child
    
    uniform = choose one or the other with bias
    point = pick position(s) in genome and only take one parent's, then switch to 
            other upon reaching the next point
    
    Can produce up two two children? I don't get it but my guess is...
    0 = Child ended up being exact same as parent and is thus discarded
    1 = Created a child based on some slections
    2 = The opposite of child 1, the 'leftovers'

    Roulette Wheel Selection: Randomly select an individual, with their
        fitness value representing their chance to be selected, out of
        the total population (Imaging a wheel with more fitness meaning
        you get a bigger slice of said wheel)
    Elitist Selection
    """
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other

        #total fitness = self.fitness + other.fitness

        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):

                #If positon above is valid and position above is a 'T' or 'I'
                    #if current position is air or breakable block or enemy:
                        #if other.position is a 'I' or 'T'
                            #new_genome.position = other.position
                        #else if other.position is solid
                            #new_genome.position = other.position
                        #else:
                            #child is invalid OR new_genome.poosition = 'I'

                new_genome[y][x] = random.choice([self.genome[y][x], other.genome[y][x]])

                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
                pass
        # do mutation; note we're returning a one-element tuple here
        self.mutate(new_genome)

        return (Individual_Grid(new_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        # New Individual Guidelines
        # 
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            # Changing meaningfulJumpVariance from 0.5 to 0.7
            meaningfulJumpVariance=0.7,
            decorationPercentage=0.5,
            leniency=0.1,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # Our own changes:
        # Here we are adding a penalty for pipes that are too tall
        for de in self.genome:
            if de[1] == "7_pipe" and de[2] > 6:
                penalties -= 3
            elif de[1] == "7_pipe" and de[2] > 4:
                penalties -= 1
        # Here, the penalty applies for too many enemies
        if len(list(filter(lambda de: de[1] == "2_enemy", self.genome))) > 8:
            penalties -= 2
        for de in self.genome:
            if de[1] == "0_hole" and de[2] > 5:
                penalties -= 2
        if len(list(filter(lambda de: de[1] == "0_hole", self.genome))) < 8:
            penalties -= 3
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                # Here, we are adding the chance to add a pipe given a large
                # enough hole
                if w >= 6:
                    new_genome.append((x, "7_pipe", random.randint(1, 3)))

                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_DE


""""""
def generate_successors(population):
    #print("Generating successors")
    #rangen = random.seed()
    results = []
    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    
    total_successors = math.floor(len(population)/2)
    fitness_values = {} 
    fitness_values_roulette = []
    total_fitness = 0

    #Put each individual in the 
    for individual in population:
        indiv_fitness = Individual.calculate_fitness(individual)
        total_fitness += indiv_fitness.fitness()
        fitness_values[individual] = indiv_fitness.fitness()
    
    #Elitism slection (Top half)
    #fitness_values.sort(fitness_values.items(), key=lambda x: x[1], reverse=True)

    sorted_fitness_list = sorted(fitness_values.items(), key=lambda x: x[1], reverse=True)
    sorted_fitness_dict = dict(sorted_fitness_list)

    #Get top half of individuals
    i = 0
    for individual in sorted_fitness_dict.keys():
        fitness_values_roulette.append(sorted_fitness_dict[individual])
        results.append(individual)
        i += 1
        if i >total_successors:
            break
    #print("Performed elitism selection")

    total_successors = math.floor(total_successors/2)
    #Roulette Selection(Random w/ fitness as weight)
    results2 = []

    #print(total_successors)
    for j in range(total_successors):
        choice = random.choices(results, fitness_values_roulette, k=1)[0]
        results2.append(choice)
        choice_index = results.index(choice)
        results.remove(results[choice_index])
        fitness_values_roulette.remove(fitness_values_roulette[choice_index])
        #results2 = random.choices(results, fitness_values_roulette, k=total_successors)
    #print("Performed roulette selection")
    results3 = []

    for parent1 in results2:
        for parent2 in results2:
            if (parent1 != parent2 and parent1.genome != [] and parent2.genome != []):
                #print("calling generate_children...")
                #print("comparing:", parent1.genome, parent2.genome)
                new_individual = Individual.generate_children(parent1, parent2)
                #print("stepping out of generate_children...")
                results3.append(new_individual[0]) #Access tuple
    

    '''
    Two solutions mayb be good in some ways and bad in others,
    combining these could lead to better one

    Elitism = compare stats directly

    1. Selection strats used to determine fitness of individuals, which ones
       shall crossover to create children
    2. Genome Crossover
    - We call generate_children from one of (or both) of the encodings,
      those being GRID and DE (Design Elements)
    - We also might modify this func, or anything that has STUDENT on it
    Mutations
    - GRID and DE also each have a mutation function, which we might modify
    - Call mutate() from generate_children(), probably


    Can individuals have multiple genomes?
    - No, each individual has one genome
    '''

    '''
    Good mario level elements:

    '''
    #print("Got successors")
    return results3


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 36
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'x') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
