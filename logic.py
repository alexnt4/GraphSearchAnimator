import pygame 
import heapq
from grap_data_cost import graph
#from graph_data3 import graph
from math import sqrt

# Constants  
display_width = 800
display_height = 800
radius = 15
speed = 3  # Frames per second

# Colors
grey = (100, 100, 100)    # Undiscovered node or edge
white = (255, 255, 255)   # Discovered edge or node outline
yellow = (200, 200, 0)    # Current node fill
red = (200, 0, 0)
black = (0, 0, 0)
blue = (50, 50, 160)
magenta = (255, 0, 255) #goal node

# Graph element parts:
# [0] : xy (position)
# [1] : adjacent node indexes 
# [2]: cost

#global
font_path = './coolvetica/esta.otf' 
font_size = 40
text_color = (255, 255, 255)  # White color
current_text = ""

def alternate_search(strategies, n, start_node, final_goal_node):
    global screen, edges, clock, current_text
    
    # Add initial colors to the graph nodes
    for element in graph:
        element.extend([grey, black])

    build_edges()
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((display_width, display_height))
    draw_graph()  # Initial drawing
    pygame.display.update()

    current_strategy_index = 0
    goal_reached = False

    while current_strategy_index < len(strategies):
        # Set the graph to black before starting the search
        for node in graph:
            node[3] = grey  # Border color
            node[4] = black  # Fill color
        for edge in edges.values():
            edge[1] = grey  # Color the edges
        draw_graph()
        pygame.display.update()

        # Wait a moment to appreciate the tree in black
        pygame.time.delay(1000)

        current_strategy = strategies[current_strategy_index]
        current_text = f"Running {current_strategy.upper()}"
        update_with_text(current_text)

        # Run the current search algorithm
        run_algorithm(current_strategy, start_node, final_goal_node, n)

        # Move to the next algorithm
        current_strategy_index += 1

        # Wait a few seconds before the next strategy
        pygame.time.delay(3000)

        # Handle events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    # Keep the window open to show the final animation
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()



def run_algorithm(algorithm, start_node, goal_node, exp):
    if algorithm == 'bfs':
        return run_bfs(start_node, goal_node, exp)
    elif algorithm == 'dfs':
        return run_dfs(start_node, goal_node, exp)
    elif algorithm == 'ucs':
        return run_ucs(start_node, goal_node, exp)
    elif algorithm == 'greedy':
        return run_greedy_best_first(start_node, goal_node, exp)
    elif algorithm == 'dls':
        return run_dls(start_node, goal_node, exp)
    elif algorithm == 'ids':
        return run_ids(start_node, goal_node, exp)
    else:
        raise ValueError(f"Algorithm {algorithm} is not supported.")


def draw_text(text, font_path, font_size, color, x, y):
    font = pygame.font.Font(font_path, font_size)  # Load the font
    text_surface = font.render(text, True, color)  # Render the text
    screen.blit(text_surface, (x, y))  # Draw the text on the screen
    #pygame.display.update()


# Function to update the screen with both graph and text
def update_with_text(text):
    global clock
    
    # Draw the graph
    draw_graph()
    # Draw the text on the screen (with specific coordinates)
    draw_text(text, font_path,font_size, text_color, 10, 10)  # Font size 36, white text, position (10, 10)
    # Update the display and control the frame rate
    pygame.display.update()
    clock.tick(speed)
        
# Normalize edge id for either order
def edge_id(n1, n2):
    return tuple(sorted((n1, n2)))  ### (1,0) y (0, 1) =(0,1)



# Build edges for visualization
def build_edges():
    global edges 
    edges = {}  # edgeid: [(n1,n2), color]
    for n1, (_, adjacents, _, _, _) in enumerate(graph):
        for n2 in adjacents:
            eid = edge_id(n1, n2)
            if eid not in edges:
                edges[eid] = [(n1, n2), grey]

# Draw the graph (nodes and edges)
def draw_graph():
    global graph, screen, edges  

    screen.fill((0, 0, 0))  # Clear the screen with black

    # Draw edges
    for e in edges.values():
        (n1, n2), color = e
        pygame.draw.line(screen, color, graph[n1][0], graph[n2][0], 2)
    
    # Draw nodes
    for xy, _,_, lcolor, fcolor in graph:
        circle_fill(xy, lcolor, fcolor, 15, 2)

# Update the screen
def update():
    global clock, current_text
    draw_graph()
    draw_text(current_text, font_path, font_size, text_color, 10, 10)
    pygame.display.update()
    clock.tick(speed)

# Draw filled circles (nodes)
def circle_fill(xy, line_color, fill_color, radius, thickness):
    global screen
    pygame.draw.circle(screen, line_color, xy, radius)
    pygame.draw.circle(screen, fill_color, xy, radius - thickness)

# Algorithms

# BFS algorithm with expansion limit
def run_bfs(start_node, goal_node, num_expansions):
    queue = [start_node]  # Start with the provided start_node
    visited = set()  # Track visited nodes
    expansions = 0  # Count the number of expansions

    while queue and expansions < num_expansions:
        n1 = queue.pop(0)  # Dequeue the first node (BFS uses a queue)
        if n1 in visited:
            continue
        visited.add(n1)
        expansions += 1  # Increment expansions counter<

        current = graph[n1]
        current[3] = white  # Update node outline color
        current[4] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[4] = magenta # Mark goal node as complete
            update()
            print(f"Goal node {goal_node} reached after {expansions} expansions.")
            return
            
        
        # Traverse adjacent nodes (BFS: add to queue)
        for n2 in current[1]:
            if graph[n2][4] == black and n2 not in visited:  # If undiscovered
                queue.append(n2)  # Enqueue node
                # Discovered n2, update colors for node and edge
                graph[n2][3] = white
                graph[n2][4] = red 
                edges[edge_id(n1, n2)][1] = white  
                update() 
        
        # Mark current node as complete
        current[4] = blue 
        update()
    
    # If we finish the loop without reaching the goal node
    if expansions >= num_expansions:
        print(f"Stopped after {expansions} expansions. Goal node {goal_node} not reached.")
    elif not queue:
        print(f"Search complete. Goal node {goal_node} not found in the graph.")


# DFS algorithm with expansion limit
def run_dfs(start_node, goal_node, num_expansions):
    stack = [start_node]  # Start with the provided start_node
    visited = set()  # Track visited nodes
    expansions = 0  # Count the number of expansions

    while stack and expansions < num_expansions:
        n1 = stack.pop()  # Pop the last added node (DFS uses stack)
        if n1 in visited:
            continue
        visited.add(n1)
        expansions += 1  # Increment expansions counter

        current = graph[n1]
        current[3] = white  # Update node outline color
        current[4] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[4] = magenta  # Mark goal node as complete
            update()
            print(f"Goal node {goal_node} reached after {expansions} expansions.")
            return
        
        # Traverse adjacent nodes in reverse order (DFS: add to stack)
        for n2 in reversed(current[1]):  # Reverse the order of adjacent nodes
            if graph[n2][4] == black and n2 not in visited:  # If undiscovered
                stack.append(n2)  # Push node to stack
                # Discovered n2, update colors for node and edge
                graph[n2][3] = white
                graph[n2][4] = red 
                edges[edge_id(n1, n2)][1] = white  
                update() 
        
        # Mark current node as complete
        current[4] = blue 
        update()

    # If we finish the loop without reaching the goal node
    if expansions >= num_expansions:
        print(f"Stopped after {expansions} expansions. Goal node {goal_node} not reached.")
    elif not stack:
        print(f"Search complete. Goal node {goal_node} not found in the graph.")


def run_ucs(start_node, goal_node, num_exp):  
    # Priority queue where elements are (cost, node)
    priority_queue = [(0, start_node)]  # Initialize the priority queue with the start node
    visited = set()  # Set to track visited nodes
    costs = {start_node: 0}  # Dictionary to store the minimum cost for each node
    expansions = 0  # Counter for the number of expansions performed

    while priority_queue and expansions < num_exp:
        # Extract the node with the lowest cost
        current_cost, n1 = heapq.heappop(priority_queue)

        # If the node has already been visited, ignore it
        if n1 in visited:
            continue

        # Mark as visited and increment the expansion counter
        visited.add(n1)
        expansions += 1

        current = graph[n1]  # Information of the current node
        adjacents = current[1]  # Adjacent nodes
        current[3] = white  # Change the border color of the node
        current[4] = yellow  # Change the fill color of the node
        update()  # Update the animation

        # Check if the goal node has been reached
        if n1 == goal_node:
            current[4] = magenta  # Highlight the goal node
            update()
            print(f"Goal node {goal_node} reached after {expansions} expansions.")
            return

        # Expand the adjacent nodes
        for n2 in adjacents:
            # Get the edge cost using the get_edge_cost function
            edge_cost = get_edge_cost(n1, n2)
            if edge_cost is None:
                continue  # If there is no edge between n1 and n2, continue with the next one

            new_cost = current_cost + edge_cost  # New accumulated cost

            # If n2 has not been visited or a cheaper path is found, update
            if n2 not in visited and (n2 not in costs or new_cost < costs[n2]):
                costs[n2] = new_cost  # Update the cost of n2
                heapq.heappush(priority_queue, (new_cost, n2))  # Add to the priority queue

                # Discovery of node n2, update colors for the node and edge
                graph[n2][3] = white  # Change the border color of the node
                graph[n2][4] = red    # Change the fill color of the node
                edges[edge_id(n1, n2)][1] = white  # Update the color of the edge
                update()  # Update the animation

        # Mark the current node as completed
        current[4] = blue
        update()

    # If the expansion limit is reached without finding the goal
    if expansions >= num_exp:
        print(f"Stopped after {expansions} expansions. Goal node {goal_node} not reached.")
    elif not priority_queue:
        print(f"Search complete. Goal node {goal_node} not found in the graph.")




def get_edge_cost(n1, n2): 
    # Normalize the edge using edge_id to ensure the order doesn't matter
    edge = edge_id(n1, n2)
    
    # Traverse the graph to find the node that contains the edge
    for node_index, node in enumerate(graph):
        adjacents = node[1]  # List of adjacent nodes
        costs = node[2]      # List of costs
        
        # If the edge is in the current node's list of adjacents
        if node_index == n1:
            if n2 in adjacents:
                # Return the cost of the edge
                return costs[adjacents.index(n2)]
    
    # If the edge is not found, return None (or you can raise an exception)
    return None



# Greedy Best-First Search algorithm
def run_greedy_best_first(start_node, goal_node, exp):
    # Priority queue where elements are (heuristic, node)
    ##current_text = "Running titil Greedy Best-First"
    priority_queue = [(heuristic(start_node, goal_node), start_node)]
    visited = set()  # Track visited nodes
    expansions = 0

    while priority_queue and expansions < exp:
        _, n1 = heapq.heappop(priority_queue)  # Pop the node with the lowest heuristic value
        if n1 in visited:
            continue
        visited.add(n1)
        expansions += 1
        current = graph[n1]
        current[3] = white  # Update node outline color
        current[4] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[4] = magenta  # Mark goal node as complete
            update()
            ##draw_text("Running Greedy Best-First", font_path, font_size, text_color, 10, 10)
            break

        # Traverse adjacent nodes (Greedy: expand based on heuristic)
        for n2 in current[1]:
            if n2 not in visited:
                heapq.heappush(priority_queue, (heuristic(n2, goal_node), n2))  # Push to the priority queue
                # Discovered n2, update colors for node and edge
                graph[n2][3] = white
                graph[n2][4] = red 
                edges[edge_id(n1, n2)][1] = white  
                update()

        
        # Mark current node as complete
        current[4] = blue 
        update_with_text(current_text)
        
# Heuristic function (Euclidean distance)
def heuristic(node, goal_node):
    x1, y1 = graph[node][0]
    x2, y2 = graph[goal_node][0]
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def run_dls(start_node, goal_node, depth_limit):  
    stack = [(start_node, 0)]  # Stack with tuples (node, depth)
    visited = set()  # Set to track visited nodes

    while stack:
        n1, depth = stack.pop()  # Get the current node and its depth
        if n1 in visited:
            continue
        visited.add(n1)

        current = graph[n1]
        current[3] = white  # Update node outline color
        current[4] = yellow  # Update node fill color
        update()

        # Check if we reached the goal node
        if n1 == goal_node:
            current[4] = magenta  # Mark goal node as complete
            update()
            print(f"Goal node {goal_node} reached at depth {depth}.")
            return

        # If depth limit is reached, do not expand further
        if depth < depth_limit:
            # Add adjacent nodes to the stack with increased depth
            for n2 in reversed(current[1]):  # Reverse the order of adjacent nodes
                if graph[n2][4] == black and n2 not in visited:
                    stack.append((n2, depth + 1))  # Push node with updated depth
                    graph[n2][3] = white  # Update outline color
                    graph[n2][4] = red    # Update fill color
                    edges[edge_id(n1, n2)][1] = white  # Update edge color
                    update()

        # Mark current node as complete
        current[4] = blue
        update()

    # If the stack is empty and the goal is not reached
    print(f"Goal node {goal_node} not found within depth limit {depth_limit}.")


def run_ids(start_node, goal_node, max_depth):
    for depth_limit in range(max_depth + 1):  # Iterate over depth limits from 0 to max_depth
        print(f"Exploring with depth limit {depth_limit}")
        
        stack = [(start_node, 0)]  # Stack with tuples (node, depth)
        visited = set()  # Reset visited nodes for each depth limit

        while stack:
            n1, depth = stack.pop()  # Get the current node and its depth

            current = graph[n1]
            current[3] = white  # Update node outline color
            current[4] = yellow  # Update node fill color
            update()

            # Check if we reached the goal node
            if n1 == goal_node:
                current[4] = magenta  # Mark goal node as complete
                update()
                print(f"Goal node {goal_node} reached at depth {depth}.")
                return

            # If depth limit is reached, do not expand further
            if depth < depth_limit:
                # Add adjacent nodes to the stack with increased depth
                for n2 in reversed(current[1]):  # Reverse the order of adjacent nodes
                    if n2 not in visited:
                        stack.append((n2, depth + 1))  # Push node with updated depth
                        graph[n2][3] = white  # Update outline color
                        graph[n2][4] = red    # Update fill color
                        edges[edge_id(n1, n2)][1] = white  # Update edge color
                        update()

            # Mark the current node as complete
            current[4] = blue
            visited.add(n1)  # Mark the node as visited
            update()

        # If the stack is empty and the goal is not reached for this depth
        print(f"Depth limit {depth_limit} reached. Goal node {goal_node} not found.")
    
    # If no solution is found within max_depth
    print(f"Goal node {goal_node} not found within maximum depth {max_depth}.")





strategies = ['bfs', 'dfs', 'ucs', 'greedy', 'dls','ids']  # List of algorithms
#strategies = ['ids']
n = 13 # Switch algorithm after reaching depth of n

##alternate_search_with_depth(strategies, n, start_node=0, final_goal_node=11)
#print(f"el nodo es: {find_goal_node_from_depth(8, n)}")

def main():
    alternate_search(strategies, n, start_node=0, final_goal_node=11)
    #print(heuristic(0,11))
    #cost = get_edge_cost(1, 0)
    #print(cost) 

if __name__ == "__main__":
    main()
