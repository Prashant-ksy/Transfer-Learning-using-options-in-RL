"""
Four Room Gridworld Environment with Agent-Space Sensors
Supports configurable room layouts and door positions for transfer learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, Dict, List, Optional
from collections import deque


class FourRoomEnv:
    """
    Four-room gridworld with light sensors for agent-space learning.
    
    Agent-space: 12-dimensional sensor readings (4 directions × 3 colors)
        - Red sensors: Detect doors
        - Green sensors: Detect goal
        - Blue sensors: Detect walls
    
    Problem-space: (room_id, x, y, has_key, door_open) - varies per instance
    """
    
    # Color channels
    RED = 0    # Doors
    GREEN = 1  # Goal
    BLUE = 2   # Walls
    
    # Directions
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    
    # Actions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, 
                 room_size: int = 5,
                 door_positions: Optional[Dict] = None,
                 goal_room: Tuple[int, int] = (1, 1),
                 sensor_range: float = 10.0,
                 seed: Optional[int] = None):
        """
        Initialize 4-room environment.
        
        Args:
            room_size: Size of each room (square)
            door_positions: Dict mapping (room1, room2) to door position
            goal_room: (row, col) of goal room in 2x2 grid
            sensor_range: Maximum distance for sensor detection
            seed: Random seed for reproducibility
        """
        self.room_size = room_size
        self.grid_size = room_size * 2 + 1  # 2x2 rooms + walls
        self.sensor_range = sensor_range
        self.goal_room = goal_room
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize door positions
        if door_positions is None:
            self.door_positions = self._generate_random_doors()
        else:
            self.door_positions = door_positions
        
        # State
        self.agent_pos = None
        self.goal_pos = None
        self.current_room = None
        
        # Memory for preventing oscillation
        self.door_history = deque(maxlen=5)
        
        # Metrics
        self.step_count = 0
        self.episode_reward = 0
        
        self.reset()
    
    def _generate_random_doors(self) -> Dict:
        """Generate random door positions between rooms."""
        doors = {}
        
        # Horizontal doors (between vertically adjacent rooms)
        for col in [0, 1]:
            # Door in wall between rooms (0,col) and (1,col)
            door_pos = np.random.randint(1, self.room_size)
            doors[((0, col), (1, col))] = ('h', door_pos)
        
        # Vertical doors (between horizontally adjacent rooms)
        for row in [0, 1]:
            # Door in wall between rooms (row,0) and (row,1)
            door_pos = np.random.randint(1, self.room_size)
            doors[((row, 0), (row, 1))] = ('v', door_pos)
        
        return doors
    
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        # Place agent in room (0, 0) at random position
        self.current_room = (0, 0)
        room_offset = self._get_room_offset(self.current_room)
        self.agent_pos = (
            room_offset[0] + np.random.randint(1, self.room_size),
            room_offset[1] + np.random.randint(1, self.room_size)
        )
        
        # Place goal in specified room at random position
        goal_offset = self._get_room_offset(self.goal_room)
        self.goal_pos = (
            goal_offset[0] + np.random.randint(1, self.room_size),
            goal_offset[1] + np.random.randint(1, self.room_size)
        )
        
        self.door_history.clear()
        self.step_count = 0
        self.episode_reward = 0
        
        return self._get_observation()
    
    def _get_room_offset(self, room: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid offset for a room."""
        row, col = room
        return (row * (self.room_size + 1), col * (self.room_size + 1))
    
    def _get_current_room(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Determine which room a position is in."""
        row = 0 if pos[0] <= self.room_size else 1
        col = 0 if pos[1] <= self.room_size else 1
        return (row, col)
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute action in environment."""
        
        Returns:
            observation: Dict with 'agent_space' and 'problem_space'
            reward: Scalar reward
            done: Whether episode is complete
            info: Additional information
        """
        old_pos = self.agent_pos
        new_pos = self._get_new_position(self.agent_pos, action)
        
        # Check if move is valid
        if self._is_valid_move(old_pos, new_pos):
            self.agent_pos = new_pos
            
            # Check if moved through door
            if self._is_door_crossing(old_pos, new_pos):
                direction = self._get_direction(action)
                self.door_history.append(direction)
                self.current_room = self._get_current_room(new_pos)
        
        # Calculate reward
        reward = -1  # Step penalty
        done = False
        
        if self.agent_pos == self.goal_pos:
            reward = 1000
            done = True
        
        self.step_count += 1
        self.episode_reward += reward
        
        # Episode timeout
        if self.step_count >= 1000:
            done = True
        
        observation = self._get_observation()
        info = {
            'step_count': self.step_count,
            'current_room': self.current_room,
            'episode_reward': self.episode_reward
        }
        
        return observation, reward, done, info
    
    def _get_new_position(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Calculate new position after action."""
        if action == self.UP:
            return (pos[0] - 1, pos[1])
        elif action == self.DOWN:
            return (pos[0] + 1, pos[1])
        elif action == self.LEFT:
            return (pos[0], pos[1] - 1)
        elif action == self.RIGHT:
            return (pos[0], pos[1] + 1)
        return pos
    
    def _is_valid_move(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> bool:
        """Check if move is valid (not into wall)."""
        # Out of bounds
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return False
        
        # Check walls
        if self._is_wall(new_pos):
            # Check if there's a door here
            return self._is_door(old_pos, new_pos)
        
        return True
    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall."""
        # Horizontal wall
        if pos[0] == self.room_size:
            return True
        # Vertical wall
        if pos[1] == self.room_size:
            return True
        return False
    
    def _is_door(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> bool:
        """Check if there's a door between old and new position."""
        old_room = self._get_current_room(old_pos)
        
        # Determine which wall we're crossing
        if new_pos[0] == self.room_size:  # Horizontal wall
            # Moving down from room (0, col) to room (1, col)
            col = 0 if old_pos[1] < self.room_size else 1
            room_pair = ((0, col), (1, col))
            
            if room_pair in self.door_positions:
                orientation, door_pos = self.door_positions[room_pair]
                if orientation == 'h':
                    # Door is at column position door_pos in the room
                    room_offset = self._get_room_offset(old_room)
                    door_col = room_offset[1] + door_pos
                    return new_pos[1] == door_col
        
        elif new_pos[1] == self.room_size:  # Vertical wall
            # Moving right from room (row, 0) to room (row, 1)
            row = 0 if old_pos[0] < self.room_size else 1
            room_pair = ((row, 0), (row, 1))
            
            if room_pair in self.door_positions:
                orientation, door_pos = self.door_positions[room_pair]
                if orientation == 'v':
                    # Door is at row position door_pos in the room
                    room_offset = self._get_room_offset(old_room)
                    door_row = room_offset[0] + door_pos
                    return new_pos[0] == door_row
        
        return False
    
    def _is_door_crossing(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int]) -> bool:
        """Check if agent crossed through a door."""
        return self._is_wall(new_pos) and self._is_door(old_pos, new_pos)
    
    def _get_direction(self, action: int) -> str:
        """Get direction name from action."""
        directions = ['N', 'E', 'S', 'W']
        return directions[action]
    
    def _get_observation(self) -> Dict:
        """Get current observation with both agent-space and problem-space."""
        
        Returns:
            Dict with keys:
                'agent_space': 12D sensor readings + 4D door history
                'problem_space': (room_id, x, y) coordinates
                'sensors': Raw 12D sensor array
                'memory': Recent door history encoding
        """
        # Agent-space: Light sensors (12D)
        sensors = self._compute_sensors()
        
        # Memory: Recent door history (one-hot encoded, 4D per door, last 2 doors)
        memory = self._encode_door_history()
        
        # Combined agent-space
        agent_space = np.concatenate([sensors, memory])
        
        # Problem-space
        problem_space = {
            'room': self.current_room,
            'position': self.agent_pos,
            'goal': self.goal_pos
        }
        
        return {
            'agent_space': agent_space,
            'problem_space': problem_space,
            'sensors': sensors,
            'memory': memory
        }
    
    def _compute_sensors(self) -> np.ndarray:
        """Compute 12D light sensor readings."""
        
        Returns:
            Array of shape (12,) with readings for:
                [0-2]: North (red, green, blue)
                [3-5]: East (red, green, blue)
                [6-8]: South (red, green, blue)
                [9-11]: West (red, green, blue)
        """
        sensors = np.zeros(12)
        
        # For each direction, compute sensor readings
        for direction in [self.NORTH, self.EAST, self.SOUTH, self.WEST]:
            base_idx = direction * 3
            
            # Red channel: Doors
            sensors[base_idx + self.RED] = self._sense_doors(direction)
            
            # Green channel: Goal
            sensors[base_idx + self.GREEN] = self._sense_goal(direction)
            
            # Blue channel: Walls
            sensors[base_idx + self.BLUE] = self._sense_walls(direction)
        
        return sensors
    
    def _sense_doors(self, direction: int) -> float:
        """Sense nearest door in given direction."""
        door_positions = []
        
        # Find all doors in the environment
        for room_pair, (orientation, pos) in self.door_positions.items():
            if orientation == 'h':
                # Horizontal door
                room_offset = self._get_room_offset(room_pair[0])
                door_grid_pos = (self.room_size, room_offset[1] + pos)
            else:
                # Vertical door
                room_offset = self._get_room_offset(room_pair[0])
                door_grid_pos = (room_offset[0] + pos, self.room_size)
            
            door_positions.append(door_grid_pos)
        
        # Find nearest door in this direction
        return self._compute_sensor_reading(door_positions, direction)
    
    def _sense_goal(self, direction: int) -> float:
        """Sense goal in given direction."""
        return self._compute_sensor_reading([self.goal_pos], direction)
    
    def _sense_walls(self, direction: int) -> float:
        """Sense nearest wall in given direction."""
        # Get all wall positions
        wall_positions = []
        
        # Horizontal wall
        for col in range(self.grid_size):
            wall_positions.append((self.room_size, col))
        
        # Vertical wall
        for row in range(self.grid_size):
            wall_positions.append((row, self.room_size))
        return self._compute_sensor_reading(wall_positions, direction)
    
    def _compute_sensor_reading(self, target_positions: List[Tuple[int, int]], 
                                  direction: int) -> float:
        """Compute sensor reading for targets in given direction."""
        
        Reading = 1.0 / (1 + distance) for nearest target in direction
        """
        min_distance = float('inf')
        
        for target_pos in target_positions:
            # Check if target is in the specified direction
            if self._is_in_direction(self.agent_pos, target_pos, direction):
                distance = self._manhattan_distance(self.agent_pos, target_pos)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf') or min_distance > self.sensor_range:
            return 0.0
        
        # Sensor response: 1.0 when on top, decays with distance
        return 1.0 - (min_distance / self.sensor_range)
    
    def _is_in_direction(self, from_pos: Tuple[int, int], 
                         to_pos: Tuple[int, int], direction: int) -> bool:
        """Check if to_pos is in direction from from_pos."""
        if direction == self.NORTH:
            return to_pos[0] < from_pos[0] and abs(to_pos[1] - from_pos[1]) <= 2
        elif direction == self.SOUTH:
            return to_pos[0] > from_pos[0] and abs(to_pos[1] - from_pos[1]) <= 2
        elif direction == self.EAST:
            return to_pos[1] > from_pos[1] and abs(to_pos[0] - from_pos[0]) <= 2
        elif direction == self.WEST:
            return to_pos[1] < from_pos[1] and abs(to_pos[0] - from_pos[0]) <= 2
        return False
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _encode_door_history(self) -> np.ndarray:
        """Encode recent door history as one-hot vectors."""
        
        Returns:
            Array of shape (8,) encoding last 2 doors (4D each)
        """
        encoding = np.zeros(8)
        direction_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        
        # Encode last 2 doors
        for i, direction in enumerate(list(self.door_history)[-2:]):
            if direction in direction_map:
                encoding[i * 4 + direction_map[direction]] = 1.0
        
        return encoding
    
    def render(self, mode='human'):
        """Visualize the environment."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw rooms
        for row in range(2):
            for col in range(2):
                offset = self._get_room_offset((row, col))
                rect = Rectangle(offset[::-1], self.room_size, self.room_size,
                               linewidth=2, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
        
        # Draw walls
        ax.axhline(y=self.room_size, color='black', linewidth=3)
        ax.axvline(x=self.room_size, color='black', linewidth=3)
        
        # Draw doors (as green squares)
        for room_pair, (orientation, pos) in self.door_positions.items():
            if orientation == 'h':
                room_offset = self._get_room_offset(room_pair[0])
                door_pos = (self.room_size, room_offset[1] + pos)
            else:
                room_offset = self._get_room_offset(room_pair[0])
                door_pos = (room_offset[0] + pos, self.room_size)
            
            ax.plot(door_pos[1], door_pos[0], 'gs', markersize=15, label='Door' if room_pair == list(self.door_positions.keys())[0] else '')
        
        # Draw agent
        ax.plot(self.agent_pos[1], self.agent_pos[0], 'bo', markersize=20, label='Agent')
        
        # Draw goal
        ax.plot(self.goal_pos[1], self.goal_pos[0], 'r*', markersize=25, label='Goal')
        
        # Draw sensor rays (for visualization)
        self._draw_sensors(ax)
        
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Four Room Environment (Step: {self.step_count})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if mode == 'human':
            plt.show()
        
        return fig
    
    def _draw_sensors(self, ax):
        """Draw sensor readings as colored rays."""
        colors = ['red', 'green', 'blue']
        directions = [(−1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
        
        obs = self._get_observation()
        sensors = obs['sensors'].reshape(4, 3)
        
        for dir_idx, (dy, dx) in enumerate(directions):
            for color_idx, color in enumerate(colors):
                intensity = sensors[dir_idx, color_idx]
                if intensity > 0.1:
                    # Draw ray
                    end_x = self.agent_pos[1] + dx * intensity * 3
                    end_y = self.agent_pos[0] + dy * intensity * 3
                    ax.arrow(self.agent_pos[1], self.agent_pos[0],
                           end_x - self.agent_pos[1], end_y - self.agent_pos[0],
                           color=color, alpha=0.3, head_width=0.3, length_includes_head=True)
    
    def get_door_positions(self) -> Dict:
        """Return current door configuration."""
        return self.door_positions.copy()
    
    def set_door_positions(self, door_positions: Dict):
        """Set door positions (for creating varied environments)."""
        self.door_positions = door_positions


if __name__ == '__main__':
    # Test the environment
    print("Testing Four Room Environment...")
    
    # Create environment with random doors
    env = FourRoomEnv(room_size=5, seed=42)
    
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Door positions: {env.door_positions}")
    
    # Reset and get observation
    obs = env.reset()
    print(f"\nAgent-space shape: {obs['agent_space'].shape}")
    print(f"Sensor readings shape: {obs['sensors'].shape}")
    print(f"Memory shape: {obs['memory'].shape}")
    print(f"\nInitial sensor readings:\n{obs['sensors'].reshape(4, 3)}")
    
    # Take random actions
    print("\nTaking 10 random actions...")
    for i in range(10):
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}, Room={info['current_room']}")
        
        if done:
            print("Episode finished!")
            break
    
    # Visualize
    env.render()
    
    print("\n✓ Environment test complete!")
