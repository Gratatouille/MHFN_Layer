import torch

def softmax(x, beta=0.01):
    e_x = torch.exp(beta * (x - torch.max(x)))
    return e_x / torch.sum(e_x)

def log_sum_exp(x, beta=0.01):
    max_val = torch.max(beta * x)
    return (beta**-1) * (torch.log(torch.sum(torch.exp(beta * x - max_val))) + max_val)

class ModernHopfieldNetwork:
    """Base class for our (Modern) Hopfield Network Layer"""

    def __init__(self, patterns_dict, beta = 0.01):
        """Initialises the Hopfield network with a set of patterns."""
        self.beta = beta
        self.patterns_dict = patterns_dict
        self.pattern_names = [i for i in patterns_dict]
        self.patterns = [torch.tensor(patterns_dict[i] , dtype=torch.float32) for i in patterns_dict]
        self.pattern_shape = patterns_dict[self.pattern_names[0]].shape
        self.N_neurons = self.pattern_shape[0] * self.pattern_shape[1]
        self.N_patterns = len(self.patterns)
        self.flat_patterns = [torch.flatten(self.patterns[i]) for i in range(len(self.patterns))]
        self.X = torch.stack(self.flat_patterns)
        self.weights = (self.X.T @ self.X) / self.N_neurons
        self.set_state(random=True)
        
        return

    def set_state(self, state=None, random=False, batch_size=1):
        if random:
            self.state = (torch.randint(0,2, size=(batch_size, self.N_neurons), dtype=torch.float32) * 2 - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            batch_size = state.shape[0]
            if state.dim() == 1:
                state = state.unsqueeze(0)
            self.state = torch.reshape(state, (batch_size, self.N_neurons))

    def update_state(self):
        first = torch.matmul(self.state, self.X.T) 
        self.state = torch.matmul(softmax(self.beta * first), self.X)

class MHFN_layer(torch.nn.Module, ModernHopfieldNetwork):
    def __init__(self, patterns_dict, beta = 0.01, steps = 1):
        torch.nn.Module.__init__(self)
        ModernHopfieldNetwork.__init__(self, patterns_dict, beta)
        self.steps = steps

    def forward(self, x):
        self.set_state(x)
        batch_size = x.shape[0]
        dim_1 = x.shape[2]
        dim_2 = x.shape[3]
        steps = self.steps
        for _ in range(steps):
            self.update_state()
        state = torch.reshape(self.state, (batch_size, 1, dim_1, dim_2))
        normalized_images = []
        
        for i in range(batch_size):
            img = state[i]
            img_max = img.max()
            img_min = img.min()
            diff = img_max - img_min
            if diff <= 0:
                norm_img = torch.zeros_like(img)
            else:
                norm_img = (img - img_min) / diff
            normalized_images.append(norm_img)

        normalized_state = torch.stack(normalized_images, dim=0)
        return normalized_state
    
    def to(self, device):
        super().to(device)
        self.patterns = [i.to(device) for i in self.patterns]
        self.flat_patterns = [i.to(device) for i in self.flat_patterns]
        self.X = self.X.to(device)
        self.weights = self.weights.to(device)
        