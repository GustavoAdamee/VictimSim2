# EXPLORER AGENT
### It walks in the environment looking for victims.

# import sys
# import os
import random
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from numpy.random import choice
from dijkstra import Dijkstra
import heapq

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class EdgeManager:
    def __init__(self):
        self.edges = {}

    def add_edge(self, node1, node2, cost):
        if node1 not in self.edges:
            self.edges[node1] = {}
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node1][node2] = cost
        self.edges[node2][node1] = cost

    def check_edge(self, node1, node2):
        return node1 in self.edges and node2 in self.edges[node1]

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 3

    def __init__(self, env, config_file, resc, priorities_vector):
        """ Construtor do agente [inserir algoritimo de busca]
        @param env: a reference to the environment
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements (for returning to the base)
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the beginning
        self.visited = set()       # to keep track of visited positions
        self.resc = resc           # reference to the rescuer agent that will be invoked when exploration finishes
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim (the victim id),(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visited.add((self.x, self.y))
        self.back_plan = []        # the plan to come back to the base
        self.edge_manager = EdgeManager()
        self.movements = priorities_vector
        self.graph = {}

    def add_edge(self, node1, node2, cost):
        if node1 not in self.graph:
            self.graph[node1] = {}
        if node2 not in self.graph:
            self.graph[node2] = {}
        self.graph[node1][node2] = cost
        self.graph[node2][node1] = cost

    def dijkstra(self, start, goal):
        queue = [(0, start)]
        distances = {start: 0}
        previous_nodes = {start: None}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == goal:
                break

            for neighbor, weight in self.graph.get(current_node, {}).items():
                distance = current_distance + weight
                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        path = []
        current_node = goal
        while current_node is not None:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()
        return path


    def get_next_position(self):
        obstacles = self.check_walls_and_lim()
        cost = self.map.get_difficulty((self.x, self.y))

        for movement in self.movements:
            direction = movement
            dx, dy = Explorer.AC_INCR[direction]
            if obstacles[direction] == VS.CLEAR and (self.x + dx, self.y + dy) not in self.visited:
                return (dx, dy)

            if obstacles[direction] == VS.CLEAR and (self.x + dx, self.y + dy) in self.visited:
                cost_neighbor = self.map.get_difficulty((self.x + dx, self.y + dy))
                if dx == 0 or dy == 0:
                    cost = cost * self.COST_LINE
                    cost_neighbor = cost_neighbor * self.COST_LINE
                else:
                    cost = cost * self.COST_DIAG
                    cost_neighbor = cost_neighbor * self.COST_DIAG

                if not self.edge_manager.check_edge((self.x, self.y), (self.x + dx, self.y + dy)):
                    self.edge_manager.add_edge((self.x, self.y), (self.x + dx, self.y + dy), cost)

        direction = random.randint(0, 7)
        return Explorer.AC_INCR[direction]

    def explore(self):
        dx, dy = self.get_next_position()
        if all([(self.x + incr[0], self.y + incr[1]) in self.visited or self.check_walls_and_lim()[i] == VS.WALL or self.check_walls_and_lim()[i] == VS.END for i, incr in Explorer.AC_INCR.items()]):
            dx, dy = self.walk_stack.pop()
            dx = -1 * dx
            dy = -1 * dy

        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.BUMPED:
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())

        if result == VS.EXECUTED:
            if (self.x + dx, self.y + dy) not in self.visited:
                self.walk_stack.push((dx, dy))

            self.visited.add((self.x + dx, self.y + dy))
            prev_x = self.x
            prev_y = self.y
            prev_diff = self.map.get_difficulty((prev_x, prev_y))

            self.x += dx
            self.y += dy
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)

            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM and seq not in self.victims:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)

            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                prev_diff = prev_diff * self.COST_LINE
                self.add_edge((prev_x, prev_y), (self.x, self.y), difficulty)
                difficulty = difficulty / self.COST_LINE
            else:
                prev_diff = prev_diff * self.COST_DIAG
                self.add_edge((prev_x, prev_y), (self.x, self.y), difficulty)
                difficulty = difficulty / self.COST_DIAG

            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

        return

    def come_back(self):
        if not self.back_plan:
            path = self.dijkstra((self.x, self.y), (0, 0))
            self.back_plan = [(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]) for i in range(len(path) - 1)]

        if self.back_plan:
            dx, dy = self.back_plan.pop(0)
            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()

            if result == VS.EXECUTED:
                self.x += dx
                self.y += dy
                self.walk_time = self.walk_time + (rtime_bef - rtime_aft)

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = (2 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ) + 100

        # keeps exploring while there is enough time
        if self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            print("Cabo o tempo de exploracao")
            self.resc.sync_explorers(self.map, self.victims)
            return False

        # proceed to the base
        self.come_back()
        return True