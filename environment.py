import numpy as np

class Santorini:
    def __init__(self, board_dim = (5,5), starting_parts = np.array([0,22,18,14,18])):
        #action_space: 2 workers * 8 moves * 8 builds = 128 options
        #moves/builds: q,w,e,a,d,z,x,c
        self.workers = [-1,-2]
        self.moves = self.builds = ['q','w','e','a','d','z','x','c']
        #board[buildings/workers, vertical, horizontal]
        self.ktoc = {'q':(-1,-1),
                     'w':(-1,0),
                     'e':(-1,1),
                     'a':(0,-1),
                     'd':(0,1),
                     'z':(1,-1),
                     'x':(1,0),
                     'c':(1,1)}
        self.itoa = [(w,m,b) for w in self.workers for m in self.moves for b in self.builds]
        self.atoi = {action:index for index,action in enumerate(self.itoa)}
        self.action_dim = len(self.itoa)
        
        self.reset(board_dim, starting_parts)
        self.state_dim_flat = len(self.get_state())
        self.state_dim = self.get_state(flattened=False).shape

    def reset(self, board_dim = (5,5), starting_parts=np.array([0,22,18,14,18])):
        #turn counter
        self.turns = 0
        
        #building pieces
        #floor, base, mid, top, dome
        self.starting_parts = starting_parts
        
        #keep track of players
        #-1, 0 , 1 for player 1, blank, player 2
        self.current_player = -1
        
        #three layers: buildings, workers, building parts
        self.board_dim = board_dim
        self.board = np.zeros((3, board_dim[0], board_dim[1]), dtype=np.int64)
        np.fill_diagonal(self.board[2,:,:],np.array(self.starting_parts))
        
        self.board[1,0,2], self.board[1,4,2] = -1, -2 #negative workers for player 1
        self.board[1,2,0], self.board[1,2,4] =  1, 2 #positive workers for player 2
        
        return(self.get_state())
        
    def print_board(self):
        print(f'Buildings:\n {self.board[0,:,:]}')
        print(f'Workers:\n {self.board[1,:,:]}')
        print(f'Parts:\n {self.board[2,:,:]}')
        
    def get_state(self, flattened=True):
        #current player has negative workers; opposing player has positive workers
        sgn = -np.sign(self.current_player)
        state = self.board.copy()
        state[1,:,:]*=sgn
        if flattened: state = np.hstack([state[:2,:,:].flatten(),state[2,:,:].diagonal()])
        return(state)
    
    def score(self):
        #get position of current player's workers
        worker_idx = np.sign(self.get_state(flattened=False)[1,:,:])==-1
        #check if workers at those positions are on top
        if (self.board[0,:,:][worker_idx] == 3).any():
            reward = 1
        else:
            reward = 0
        return(reward)
        
    def move(self,worker,key):
        #worker is either -1, -2; pov of current player
        if worker not in [-1,-2]: raise ValueError('Wrong Worker')
        
        #get source and destinations
        state = self.get_state(flattened=False)
        worker_idx = np.where(state[1,:,:]==worker)
        src = (worker_idx[0][0],worker_idx[1][0])
        worker_num = self.board[1,src[0], src[1]]
        delta = self.ktoc[key]
        dest = (src[0]+delta[0],src[1]+delta[1])
        
        #check if correct turn
        if np.sign(self.board[1,src[0],src[1]])!=self.current_player:
            raise ValueError('Wrong Player')
        
        #check legality of the move; within the board, one level, no one standing
        inbound =  (-1 < dest[0] < self.board_dim[0]) & (-1 < dest[1] < self.board_dim[1])
        blank_tile = self.board[1,dest[0],dest[1]]==0
        one_level = (self.board[0,dest[0],dest[1]] - self.board[0,src[0],src[1]]) <= 1
        not_dome = self.board[0,dest[0],dest[1]] < 4
        
        if inbound & one_level & blank_tile & not_dome:
            self.board[1,src[0],src[1]] = 0
            self.board[1,dest[0],dest[1]] = worker_num
        else:
            #print(f'Illegal Move\n Inbound: {inbound}\n One Level: {one_level}\n Blank Tile: {blank_tile}')
            raise ValueError(f'Illegal Move')
    
    def build(self,worker,key):
        #worker is either -1, -2; pov of current player
        if worker not in [-1,-2]: raise ValueError('Wrong Worker')
        
        #get source and destinations
        state = self.get_state(flattened=False)
        worker_idx = np.where(state[1,:,:]==worker)
        src = (worker_idx[0][0],worker_idx[1][0])
        worker_num = self.board[1,src[0], src[1]]
        delta = self.ktoc[key]
        dest = (src[0]+delta[0],src[1]+delta[1])
        
        #check tower size legality
        to_build = self.board[0,dest[0],dest[1]] + 1
        if to_build <=4:
            parts_left = self.board[2,to_build,to_build]
        else:
            raise ValueError('Building too tall')
            
        #check if correct turn
        if np.sign(self.board[1,src[0],src[1]])!=self.current_player:
            raise ValueError('Wrong Player')
            
        #check legality of the build; within the board, enough parts, no one standing
        inbound =  (-1 < dest[0] < self.board_dim[0]) & (-1 < dest[1] < self.board_dim[1])
        enough_parts = parts_left > 0
        blank_tile = self.board[1,dest[0],dest[1]]==0
        if inbound & enough_parts & blank_tile:
            self.board[0,dest[0],dest[1]] = to_build
            self.board[2,to_build,to_build] -= 1
        #if no parts left; do nothing
        elif inbound & (not enough_parts) & blank_tile:
            pass
        else:
            #print(f'Illegal Build\n Inbound: {inbound}\n Enough Parts: {enough_parts}\n Blank Tile: {blank_tile}')
            raise ValueError('Illegal Build')
                          
    def step(self, action_idx, switch_player=True , flattened = True, move_reward = -1e-3,
            too_many_mult = 5):
        self.turns+=1
        reward = move_reward
        worker,move_key,build_key = self.itoa[action_idx]
        
        #try to move
        try:
            self.move(worker,move_key)
        except:
            next_state = self.get_state(flattened)
            reward += -1e3
            done = True
            if switch_player: self.current_player *= -1
            return(next_state,reward,done,self.current_player)
            
        #try to build
        try:
            self.build(worker,build_key)
        except:
            next_state = self.get_state(flattened)
            reward += -1e3
            done = True
            if switch_player: self.current_player *= -1
            return(next_state,reward,done,self.current_player)
        
        #check if it's already too many turns
        too_many_turns = self.turns > too_many_mult * self.starting_parts.sum()
        next_state = self.get_state(flattened)
        reward += self.score()
        done = True if (self.score()==1 or too_many_turns) else False
        if switch_player: self.current_player *= -1
        return(next_state,reward,done,self.current_player)
    
    def legal_moves(self):
        legals = []
        for i,j in enumerate(self.itoa):
            legal = True
            old_board = self.board.copy()
        
            #try to move
            try:
                self.move(j[0],j[1])
            except:
                #print(f'illegal move {i}')
                legal = False
            #try to build
            try:
                self.build(j[0],j[2])
            except:
                #print(f'illegal build {i}')
                legal = False
                
            if legal: 
                #print(f'legal {i}')
                legals.append(i)
                legal = True
                
            self.board = old_board
        return(legals)