# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

#-----------------------------------------------------------
from collections import deque

def bfs(start, goal, get_neighbors):
    """Breadth-First Search to find the shortest path from start to goal using the explored map."""
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for neighbor in get_neighbors(current):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current

    # Check if the goal was reached
    if goal not in came_from:
        return []

    # Reconstruct path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

#-----------------------------------------------------------

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

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc, priorities_vector):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.env = env
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.visited = set()       # to keep track of visited nodes
        self.stack = Stack()       # stack for DFS
        self.stack.push((self.x, self.y))  # start from the base

        # priorities for the explorer on the DFS
        # example: [2, 1, 0, 7, 6, 5, 4, 3]
        self.priorities = priorities_vector

        self.path_to_base = []

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_neighbors(self, pos):
        """Get valid neighboring positions based on the explored map."""
        x, y = pos
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        # Filter out invalid positions (e.g., not explored)
        valid_neighbors = [n for n in neighbors if self.is_valid_position(n)]
        return valid_neighbors

    def is_valid_position(self, pos):
        """Check if the position is valid based on the explored map."""
        if pos in self.map.data:
            actions_results = self.map.get_actions_results(pos)
            return VS.CLEAR in actions_results
        return False

    def get_next_position(self):
        """ Uses an online DFS to get the next position that can be explored (no wall and inside the grid). """
        while not self.stack.is_empty():
            current_pos = self.stack.pop()
            self.visited.add(current_pos)
            self.x, self.y = current_pos
            # Check the neighborhood walls and grid limits
            obstacles = self.check_walls_and_lim()
            directions = list(Explorer.AC_INCR.items())
            
            # Order the directions according to the priorities
            directions.sort(key=lambda x: self.priorities.index(x[0]))

            for direction, (dx, dy) in directions:
                if obstacles[direction] == VS.CLEAR:
                    next_pos = (self.x + dx, self.y + dy)
                    if next_pos not in self.visited:
                        self.stack.push(current_pos)  # push current position back to stack
                        self.stack.push(next_pos)  # push next position to stack
                        return Explorer.AC_INCR[direction]
        print("No more positions to explore")
        return (0, 0)  # if no more positions to explore, stay in place
        # return 0, 0  # if no more positions to explore, stay in place
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()


        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return
    
    def come_back(self):
        # if not self.path_to_base:
        #     self.path_to_base = bfs((self.x, self.y), (0, 0), self.get_neighbors)
        
        # if len(self.path_to_base) > 1:
        #     next_step = self.path_to_base[1]
        #     dx = next_step[0] - self.x
        #     dy = next_step[1] - self.y

        #     result = self.walk(dx, dy)
        #     if result == VS.BUMPED:
        #         print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
        #         return
            
        #     if result == VS.EXECUTED:
        #         # update the agent's position relative to the origin
        #         self.x += dx
        #         self.y += dy
        #         print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        #         self.path_to_base.pop(0)
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 2 * self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

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
