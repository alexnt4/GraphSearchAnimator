import pygame 
import heapq
from graph_data import graph 
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

# Graph element parts:
# [0] : xy (position)
# [1] : adjacent node indexes 
# [2] : node edge color   
# [3] : node fill color 

#global
font_path = './coolvetica/esta.otf' 
font_size = 40
text_color = (255, 255, 255)  # White color
current_text = ""


# This function is used to know which node will be the initial node of the next search, that is, it gives us
# the last node of the current search and this is sent as the initial node for the next

def find_goal_node_from_depth(start_node, depth):
    # This function traverses the nodes of the graph from a starting node
    # until reaching a given depth and returns the node with the greatest depth

    visited = set()
    queue = [(start_node, 0)]  # (node, current depth)
    
    last_node_at_depth = start_node  # Last node reached at the given depth

    while queue:
        current_node, current_depth = queue.pop(0)

        if current_depth >= depth:  # If we have reached the desired depth, stop
            break

        for adjacent_node in graph[current_node][1]:  # Get adjacent
            if adjacent_node not in visited:
                visited.add(adjacent_node)
                queue.append((adjacent_node, current_depth + 1))
                last_node_at_depth = adjacent_node  # Update the deepest node
    return last_node_at_depth

def alternate_search_with_depth(strategies, n, start_node, final_goal_node):
    global screen, edges, clock, current_text
    
    # Add initial colors to graph nodes  
    for element in graph:
        element.extend([grey, black])

    build_edges()
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((display_width, display_height))
    draw_graph()  # Initial drawing
    pygame.display.update()

    # Main loop to keep the window open and handle events
    current_strategy_index = 0
    current_node = start_node
    goal_reached = False  # Add a variable to mark if the end node was reached

    while current_strategy_index < len(strategies):
        current_strategy = strategies[current_strategy_index]
        current_text = f"Running {current_strategy.upper()}"  # We updated the text of the strategy name

        # Find the end node using depth n
        goal_node = find_goal_node_from_depth(current_node, n)

        # Render text before each search and update the screen
        update_with_text(current_text)

        # Run the search with the current algorithm and animate
        run_algorithm(current_strategy, current_node, goal_node)

        # Check if we have reached the final end node DURING the animation
        if goal_node == final_goal_node:
            goal_reached = True
            print(f"Goal node {final_goal_node} found!")
            break

        # Update the start node for the next iteration if we have not reached the end node
        current_node = goal_node

        # Switch to the next strategy
        current_strategy_index += 1

        pygame.display.update()
        pygame.time.delay(4000)  # Wait 4 seconds before next strategy

    # After the cycle ends (either by reaching the end node or by exhausting strategies)
    if goal_reached:
        print("Animation finished. Program won't close immediately.")
    else:
        print("Solution not found after exploring all strategies.")

    # Keep the program open to see the full animation
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Close window if 'x' is clicked

    pygame.quit()  # Properly close Pygame



def run_algorithm(algorithm, start_node, goal_node):
    if algorithm == 'bfs':
        return run_bfs(start_node, goal_node)
    elif algorithm == 'dfs':
        return run_dfs(start_node, goal_node)
    elif algorithm == 'ucs':
        return run_ucs(start_node, goal_node)
    elif algorithm == 'greedy':
        return run_greedy_best_first(start_node, goal_node)
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
    return tuple(sorted((n1, n2)))

# Build edges for visualization
def build_edges():
    global edges 
    edges = {}  # edgeid: [(n1,n2), color]
    for n1, (_, adjacents, _, _) in enumerate(graph):
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
    for xy, _, lcolor, fcolor in graph:
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



# BFS algorithm
def run_bfs(start_node, goal_node):
    queue = [start_node]  # Start with the provided start_node
    visited = set()  # Track visited nodes

    while queue:
        n1 = queue.pop(0)  # Dequeue the first node (BFS uses a queue)
        if n1 in visited:
            continue
        visited.add(n1)
        current = graph[n1]
        current[2] = white  # Update node outline color
        current[3] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[3] = blue  # Mark goal node as complete
            update()
            break
        
        # Traverse adjacent nodes (BFS: add to queue)
        for n2 in current[1]:
            if graph[n2][3] == black and n2 not in visited:  # If undiscovered
                queue.append(n2)  # Enqueue node
                # Discovered n2, update colors for node and edge
                graph[n2][2] = white
                graph[n2][3] = red 
                edges[edge_id(n1, n2)][1] = white  
                update() 
        
        # Mark current node as complete
        current[3] = blue 
        update()

# DFS algorithm
def run_dfs(start_node, goal_node):
    stack = [start_node]  # Start with the provided start_node
    visited = set()  # Track visited nodes

    while stack:
        n1 = stack.pop()  # Pop the last added node (DFS uses stack)
        if n1 in visited:
            continue
        visited.add(n1)
        current = graph[n1]
        current[2] = white  # Update node outline color
        current[3] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[3] = blue  # Mark goal node as complete
            update()
            break
        
        # Traverse adjacent nodes in reverse order (DFS: add to stack)
        for n2 in reversed(current[1]):  # Reverse the order of adjacent nodes
            if graph[n2][3] == black and n2 not in visited:  # If undiscovered
                stack.append(n2)  # Push node to stack
                # Discovered n2, update colors for node and edge
                graph[n2][2] = white
                graph[n2][3] = red 
                edges[edge_id(n1, n2)][1] = white  
                update() 
        
        # Mark current node as complete
        current[3] = blue 
        update()


# UCS algorithm
def run_ucs(start_node, goal_node):
    # Priority queue where elements are (cost, node)
    priority_queue = [(0, start_node)]
    visited = set()  # Track visited nodes
    costs = {start_node: 0}  # Cost dictionary to store the minimum cost to each node

    while priority_queue:
        current_cost, n1 = heapq.heappop(priority_queue)  # Pop the node with the lowest cost
        if n1 in visited:
            continue
        visited.add(n1)
        current = graph[n1]
        current[2] = white  # Update node outline color
        current[3] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[3] = blue  # Mark goal node as complete
            update()
            break

        # Traverse adjacent nodes (UCS: expand based on cost)
        for n2 in current[1]:
            edge_cost = 1  # Assuming uniform cost of 1 for each edge
            new_cost = current_cost + edge_cost
            if n2 not in visited and (n2 not in costs or new_cost < costs[n2]):
                costs[n2] = new_cost  # Update the cost for this node
                heapq.heappush(priority_queue, (new_cost, n2))  # Push to the priority queue
                # Discovered n2, update colors for node and edge
                graph[n2][2] = white
                graph[n2][3] = red 
                edges[edge_id(n1, n2)][1] = white  
                update()
        
        # Mark current node as complete
        current[3] = blue 
        update()


# Greedy Best-First Search algorithm
def run_greedy_best_first(start_node, goal_node):
    # Priority queue where elements are (heuristic, node)
    ##current_text = "Running titil Greedy Best-First"
    priority_queue = [(heuristic(start_node, goal_node), start_node)]
    visited = set()  # Track visited nodes

    while priority_queue:
        _, n1 = heapq.heappop(priority_queue)  # Pop the node with the lowest heuristic value
        if n1 in visited:
            continue
        visited.add(n1)
        current = graph[n1]
        current[2] = white  # Update node outline color
        current[3] = yellow  # Current node fill color

        # Check if we reached the goal node
        if n1 == goal_node:
            current[3] = blue  # Mark goal node as complete
            update()
            ##draw_text("Running Greedy Best-First", font_path, font_size, text_color, 10, 10)
            break

        # Traverse adjacent nodes (Greedy: expand based on heuristic)
        for n2 in current[1]:
            if n2 not in visited:
                heapq.heappush(priority_queue, (heuristic(n2, goal_node), n2))  # Push to the priority queue
                # Discovered n2, update colors for node and edge
                graph[n2][2] = white
                graph[n2][3] = red 
                edges[edge_id(n1, n2)][1] = white  
                update()

        
        # Mark current node as complete
        current[3] = blue 
        update_with_text(current_text)
        
# Heuristic function (Euclidean distance)
def heuristic(node, goal_node):
    x1, y1 = graph[node][0]
    x2, y2 = graph[goal_node][0]
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)




strategies = ['bfs', 'dfs', 'ucs', 'greedy']  # List of algorithms
n = 2  # Switch algorithm after reaching depth of n

##alternate_search_with_depth(strategies, n, start_node=0, final_goal_node=11)
#print(f"el nodo es: {find_goal_node_from_depth(8, n)}")

def main():
    alternate_search_with_depth(strategies, n, start_node=0, final_goal_node=11)

if __name__ == "__main__":
    main()