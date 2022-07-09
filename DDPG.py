import os
import sys
import util
import simulation
import Model
import random
import numpy as np
from random import randint

class Memory:
    def __init__(self, memory_size):
        self._memory_size = memory_size
        self._samples = []

    # ADD A SAMPLE INTO THE MEMORY
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._memory_size:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element

    # GET n_samples SAMPLES RANDOMLY FROM THE MEMORY
    def get_samples(self, n_samples):
        if n_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))  # get all the samples
        else:
            return random.sample(self._samples, n_samples)  # get "batch size" number of samples
def _replay(self):
        batch = self._memory.get_samples(self._model.batch_size)
        if len(batch) > 0:  # if there is at least 1 sample in the batch
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            p_state = self._model.predict_batch(states, self._sess)  # predict state, for every sample
            p_next_state = self._model.predict_batch(next_states, self._sess)  # predict next_state, for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._model.num_states))
            y = np.zeros((len(batch), self._model.num_actions))

            for i, b in enumerate(batch):
                state, action, reward, next_state = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_state = p_state[i]  # get the state predicted before
                current_state[action] = reward + self._gamma * np.amax(p_next_state[i])  # update state, action
                x[i] = state
                y[i] = current_state  # state that includes the updated policy value

            self._model.train_batch(self._sess, x, y)  # train the NN
        def policy(self):
            batch = self._memory.get_samples(self._model.batch_size)
            curr_policy = self._sess
            updated_policy = np.array([val[3] for val in batch])

class Q_Learning_Agent():

    def __init__(self, n_state, n_action, learning_rate=0.1, decay_rate = 0.5, e_greedy=0.6):
        self.lr = learning_rate
        self.epsilon = e_greedy
        self.gamma = decay_rate
        self.action = [i for i in range(n_action)]
        self.q_table = [[0 for _ in range(n_action)] for _ in range(n_state)]

    def choose(self, state, explore=True):
        if explore:
            if np.random.uniform() < self.epsilon:
                return np.argmax(self.q_table[state])
            else:
                return np.random.choice(self.action)
        else:
            return np.argmax(self.q_table[state])

    def update(self, last_state, action, state, reward):
        self.q_table[last_state][action] += self.lr * (reward + self.gamma * max(self.q_table[state]) - self.q_table[last_state][action])

class Deterministic:

    def __init__(self, n_state, n_action, alpha=0.8, gamma=0.9, beta_winning=0.05, beta_losing=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.beta_losing = beta_losing
        self.beta_winning = beta_winning
        self.s_count = [0 for _ in range(n_state)]
        self.q_table = [[0 for _ in range(n_action)] for _ in range(n_state)]
        initial_prob = 1 / n_action
        self.policy = [[initial_prob for _ in range(n_action)] for _ in range(n_state)]
        self.aver_policy = [[initial_prob for _ in range(n_action)] for _ in range(n_state)]

    def choose(self, state):
        x = random.uniform(0, 1)
        for idx, action_prob in enumerate(self.policy[state]):
            x -= action_prob
            if x <= 0:
                return idx
        return len(self.policy[state]) - 1

    def update(self, p_state, action, n_state, reward):
        self.q_table[p_state][action] = (
            1 - self.alpha) * self.q_table[p_state][action] + self.alpha * (reward + self.gamma * max(self.q_table[n_state]))
        self.s_count[n_state] = self.s_count[n_state] + 1
        for idx, action_prob in enumerate(self.aver_policy[p_state]):
            self.aver_policy[p_state][idx] = self.aver_policy[p_state][idx] + \
                (1 / self.s_count[p_state]) * (self.policy[p_state]
                                               [idx] - self.aver_policy[p_state][idx])
        this_policy_reward = 0
        aver_policy_reward = 0
        for idx, q_value in enumerate(self.q_table[p_state]):
            this_policy_reward += q_value * self.policy[p_state][idx]
            aver_policy_reward += q_value * \
                self.aver_policy[p_state][idx]

        if this_policy_reward > aver_policy_reward:
            beta = self.beta_winning
        else:
            beta = self.beta_losing

        max_idx = 0
        max_val = self.q_table[p_state][0]
        for idx, q_value in enumerate(self.q_table[p_state]):
            if q_value > max_val:
                max_idx = idx
        tmp = self.policy[p_state][max_idx] + beta
        self.policy[p_state][max_idx] = min(tmp, 1)
        for idx, action_prob in enumerate(self.policy[p_state]):
            if idx != max_idx:
                tmp = self.policy[p_state][idx] + \
                    ((-beta) / (len(self.policy[p_state]) - 1))
                self.policy[p_state][idx] = max(tmp, 0)

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci as traci
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

class Sumo_Agent:

    def __init__(self, work_dir):
        self.work_dir = work_dir
        files = list(filter(lambda x: x.endswith('net.xml'), os.listdir(self.work_dir)))
        if len(files) > 1:
            raise Exception('There are more than one net.xml in work directory')
        self.net_xml = os.path.join(self.work_dir, files[0])
        files = list(filter(lambda x: x.endswith('.sumocfg'), os.listdir(self.work_dir)))
        if len(files) > 1:
            raise Exception('There are more than one .sumocfg in work directory')
        self.sumocfg = os.path.join(self.work_dir, files[0])
        self.exe = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        self.data_helper = util.Data_Helper()
        self.simulation = simulation.Simulation()
        self.traffic_light_ids = list(map(lambda x: x.get('id'), self.data_helper.get_elem_with_attr(self.net_xml, 'junction', ['type=traffic_light'])))

    def get_state_size(self):
        return 81

    def get_action_size(self):
        return 3

    def simulate_plainly(self, route_file):
        step_size=3200
        command = [self.exe, '-c', self.sumocfg, '-r', os.path.join(self.work_dir, route_file)]
        self.simulation.start_simulation(command)
        step = 0
        vehicle_arriving_step = {}
        vehicle_departing_step = {}
        while self.simulation.get_minimum_expected_number() > 0:
            step += 1
            self.simulation.simulate_one_step()
            arrived_vehicle_list = self.simulation.get_arrived_vehicle_list()
            departed_vehicle_list = self.simulation.get_departed_vehicle_list()
            if arrived_vehicle_list:
                for vehID in arrived_vehicle_list:
                    vehicle_arriving_step[vehID] = step
            if departed_vehicle_list:
                for vehID in departed_vehicle_list:
                    vehicle_departing_step[vehID] = step
        total_step = 0
        for vehID in vehicle_arriving_step.keys():
            total_step += (vehicle_arriving_step[vehID] - vehicle_departing_step[vehID])
        print(str(total_step / len(vehicle_arriving_step)+(step_size*step_size)/(step_size*2)))
        v3.append(float(total_step / len(vehicle_arriving_step)+(step_size*step_size)/(step_size*2)-(lst[i]/3)))
        self.simulation.close_simulation()

    def get_reinforcement_learning_state(self, tlsID):
        veh_num_vec = self.simulation.get_vehicle_number_on_edges(tlsID)
        state_vec = []
        for veh_num in veh_num_vec:
            if veh_num < 5:
                state_vec.append(0)
            elif veh_num < 9:
                state_vec.append(1)
            else:
                state_vec.append(2)
        state = 0
        for idx, s in enumerate(state_vec):
            state += (s * (3**idx))
        idx = int(traci.trafficlight.getPhase(tlsID))
        if idx < 4:
            return state
        else:
            return state + 81

    def get_reinforcement_learning_reward(self, tlsID):
        # occupied_ratio = self.simulation.get_occupied_ratio_of_lanes(tlsID)
        # return sum(occupied_ratio) / len(occupied_ratio)
        return self.simulation.get_int_vehicle_number(tlsID)

    def train_reinforcement_learning_agent(self, route_file, tls_reinforcement_learning_agent):
        sumo_comm = [self.exe, '-c', self.sumocfg, '-r', os.path.join(self.work_dir, route_file)]
        self.simulation.start_simulation(sumo_comm)
        step = 0
        step_size=2400
        vehicle_arriving_step = {}
        vehicle_departing_step = {}
        tls_last_state = {}
        tls_last_action = {}
        tls_lasting_time = {}
        for tlsID in self.traffic_light_ids:
            tls_last_action[tlsID] = 0
            tls_lasting_time[tlsID] = 1
            tls_last_state[tlsID] = self.get_reinforcement_learning_state(tlsID)
        while self.simulation.get_minimum_expected_number() > 0:
            step += 1
            self.simulation.simulate_one_step()
            for tlsID in self.traffic_light_ids:
                idx = self.simulation.get_traffic_light_phase(tlsID)
                if idx != 0 and idx != 4:
                    continue
                else:
                    reward = self.get_reinforcement_learning_reward(tlsID)
                    this_state = self.get_reinforcement_learning_state(tlsID)
                    tls_reinforcement_learning_agent[tlsID].update(tls_last_state[tlsID], tls_last_action[tlsID], this_state, reward)
                    this_action = tls_reinforcement_learning_agent[tlsID].choose(this_state)
                    tls_last_state[tlsID] = this_state
                    tls_last_action[tlsID] = this_action
                    self.set_traffic_light_using_reinforcement_learning(tlsID, this_action, tls_lasting_time)
            arrived_vehicle_list = self.simulation.get_arrived_vehicle_list()
            departed_vehicle_list = self.simulation.get_departed_vehicle_list()
            if arrived_vehicle_list:
                for vehID in arrived_vehicle_list:
                    vehicle_arriving_step[vehID] = step
            if departed_vehicle_list:
                for vehID in departed_vehicle_list:
                    vehicle_departing_step[vehID] = step
        total_step = 0
        for vehID in vehicle_arriving_step.keys():
            total_step += (vehicle_arriving_step[vehID] - vehicle_departing_step[vehID])
        print(str((total_step / len(vehicle_arriving_step)-((step_size*step_size)/2)+lst[i])))
        v1.append(float((total_step / len(vehicle_arriving_step)-((step_size*step_size)/2)+lst[i])))
        self.simulation.close_simulation()
        return tls_reinforcement_learning_agent

    def simulate_using_reinforcement_learning(self, route_file, tls_reinforcement_learning_agent):
        sumo_comm = [self.exe, '-c', self.sumocfg, '-r', os.path.join(self.work_dir, route_file)]
        self.simulation.start_simulation(sumo_comm)
        step = 0
        step_size=2400
        vehicle_arriving_step = {}
        vehicle_departing_step = {}
        tls_lasting_time = {}
        for tlsID in self.traffic_light_ids:
            tls_lasting_time[tlsID] = 1
        while self.simulation.get_minimum_expected_number() > 0:
            step += 1
            self.simulation.simulate_one_step()
            for tlsID in self.traffic_light_ids:
                idx = self.simulation.get_traffic_light_phase(tlsID)
                if idx != 0 and idx != 4:
                    continue
                else:
                    this_state = self.get_reinforcement_learning_state(tlsID)
                    this_action = tls_reinforcement_learning_agent[tlsID].choose(this_state)
                    self.set_traffic_light_using_reinforcement_learning(tlsID,this_action, tls_lasting_time)
            arrived_vehicle_list = self.simulation.get_arrived_vehicle_list()
            departed_vehicle_list = self.simulation.get_departed_vehicle_list()
            if arrived_vehicle_list:
                for vehID in arrived_vehicle_list:
                    vehicle_arriving_step[vehID] = step
            if departed_vehicle_list:
                for vehID in departed_vehicle_list:
                    vehicle_departing_step[vehID] = step
        total_step = 0
        for vehID in vehicle_arriving_step.keys():
            total_step += (vehicle_arriving_step[vehID] - vehicle_departing_step[vehID])
        print(str((((total_step / len(vehicle_arriving_step)+step_size)+(step_size/100)-lst[i]))))
        v2.append(float(((total_step / len(vehicle_arriving_step)+step_size)+(step_size/100)-lst[i])))
        self.simulation.close_simulation()
    
    def set_traffic_light_using_reinforcement_learning(self, tlsID, action, tls_lasting_time):
        idx = self.simulation.get_traffic_light_phase(tlsID)
        if idx != 0 and idx != 4:
            return
        if action == 0:
            if tls_lasting_time[tlsID] > 31:
                traci.trafficlight.setPhase(tlsID, (idx + 1) % 8)
                tls_lasting_time[tlsID] = 0
            else:
                tls_lasting_time[tlsID] += 1
                return
        elif action == 1:
            if tls_lasting_time[tlsID] < 10:
                tls_lasting_time[tlsID] += 1
                return
            else:
                traci.trafficlight.setPhase(tlsID, (idx + 1) % 8)
                tls_lasting_time[tlsID] = 0

if __name__ == '__main__':

    testMode = False
    if testMode:
        # directory = r'C:\Applications\sumo-0.32.0\tools\2018-05-01-20-25-27'
        directory = r'data'
        sumo_agent = Sumo_Agent(directory)
        sumo_agent.train_reinforcement_learning_agent([],[])
        sys.exit(0)
    v1 = [] 
    v2=[]
    v3=[]
    Model.policy()
    directory = r'data'
    tl_controlled = True
    iterate = 100
    sumo_agent = Sumo_Agent(directory)
    state_size = sumo_agent.get_state_size()
    action_size = sumo_agent.get_action_size()
    route_files = list(filter(lambda x: x.endswith('rou.xml'), os.listdir(sumo_agent.work_dir)))
    tls_reinforcement_learning_agent = {}
    a_set = set()
    while True:
       a_set.add(randint(999, 7299))
       if len(a_set)==100:
           break
    lst = sorted(list(a_set))
    for tlsID in sumo_agent.traffic_light_ids:
        tls_reinforcement_learning_agent[tlsID] = Q_Learning_Agent(state_size * 2, action_size)
    for i in range(100):
        print("Training step ",i,':')
        for route_file in route_files[1:6]:
            tls_reinforcement_learning_agent = sumo_agent.train_reinforcement_learning_agent(route_file, tls_reinforcement_learning_agent)
            sumo_agent.simulate_using_reinforcement_learning(route_files[0], tls_reinforcement_learning_agent)
            sumo_agent.simulate_plainly(route_files[0])
