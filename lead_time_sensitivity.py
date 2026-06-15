import sys, os, random, csv
from scipy.stats import skew as scipy_skew
import numpy as np
import torch
 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
 
from envs.dist_center import DistCenter
from common.models import ModelA2C, AgentA2C
 
INVT_PATH  = os.path.join(ROOT, 'data', 'starting_invt_brake_pads.csv')
ORDER_PATH = os.path.join(ROOT, 'data', 'incoming_orders.csv')
MODEL_PATH = os.path.join(ROOT, 'saves', 'best', 'best_+1977.900_3692000.dat')
N_EPISODES = 200
SEED       = 42
 
# Scenarii: (nume, categorie, plane_days, freight_days, ship_days)

SCENARIOS = [
    ('Baseline',       'baseline',    2,  7, 14),
    ('Avion+1zi',      'mild',        3,  7, 14),
    ('Avion+2zile',    'mild',        4,  7, 14),
    ('Freight+3zile',  'mild',        2, 10, 14),
    ('Freight+5zile',  'mild',        2, 12, 14),
    ('Avion+Fr mild',  'severe',      3, 10, 14),
    ('Avion+Fr sever', 'severe',      4, 12, 14),
    ('Avion rapid',    'improved',    1,  7, 14),
    ('Freight rapid',  'improved',    2,  5, 14),
    ('Ambii rapizi',   'improved',    1,  5, 14),
    ('Stoch usor',     'stochastic', -1, -2, 14),
    ('Stoch sever',    'stochastic', -3, -4, 14),
]
 
# Distributii pentru lead times stochastice: cod -> (min, max) uniform
STOCH_PARAMS = {
    -1: (1, 4),   # avion usor
    -2: (5, 10),  # freight usor
    -3: (1, 5),   # avion sever
    -4: (7, 13),  # freight sever
}
 
def get_lt(code):
    #Returneaza lead time fix sau aleator
    if code > 0:
        return code
    lo, hi = STOCH_PARAMS[code]
    return int(np.random.randint(lo, hi + 1))
 
 
#  Mediu cu lead times configurabile
class PerturbedDistCenter(DistCenter):
    def __init__(self, p, f, s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p, self._f, self._s = p, f, s
 
    def reset(self):
        # Suprascriu reset() ca sa nu piarda _p, _f, _s la reinitializare
        self.__init__(self._p, self._f, self._s,
                      self.name, self.invt_path, self.order_path)
        return self.observation()
 
    def step(self, actions):
        self.step_count += 1
        p = min(get_lt(self._p), self.ship_days)
        f = min(get_lt(self._f), self.ship_days)
        s = min(get_lt(self._s), self.ship_days)
        self.update_refill_order_queue(actions[0], p)
        self.update_refill_order_queue(actions[1], f)
        self.update_refill_order_queue(actions[2], s)
        self.update_incoming_order_queue()
        self.update_previous_order_queue()
        self.update_invt_history_queue()
        next_state = self.observation()
        reward     = self.calculate_reward(actions)
        done = (self.determine_terminate(self.invt.get_invt_quantities())
                or self.step_count == 1000)
        return next_state, reward, done, self.step_count
 
 
# Incarcare agent antrenat
def load_agent():
    env = DistCenter(name='Charlotte', invt_path=INVT_PATH, order_path=ORDER_PATH)
    net = ModelA2C(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    net.eval()
    return AgentA2C(net, device='cpu')
 
 
# Rulare un scenariu, returneaza dict cu metrici 
def run_scenario(name, category, p, f, s, agent):
    rewards, steps, ship_actions = [], [], []
 
    for ep in range(N_EPISODES):
        random.seed(SEED + ep)
        np.random.seed(SEED + ep)
 
        env = PerturbedDistCenter(p, f, s,
                                  name='Charlotte',
                                  invt_path=INVT_PATH,
                                  order_path=ORDER_PATH)
        obs, total = env.observation(), 0.0
 
        while True:
            action_flat, _ = agent([obs], [None])
            ap  = float(action_flat[0][0])
            af  = float(action_flat[0][1])
            ash = float(action_flat[0][2])
            obs, reward, done, _ = env.step([ap, af, ash])
            total += reward
            ship_actions.append(ash)
            if done:
                break
 
        rewards.append(total)
        steps.append(env.step_count)
 
    r   = np.array(rewards)
    s_a = np.array(steps)
    sh  = np.array(ship_actions)
 
    cv   = float(sh.std() / sh.mean() * 100) if sh.mean() > 1e-9 else 0.0
    skew = float(scipy_skew(sh)) if len(sh) > 2 else 0.0
 
    return {
        'scenario'       : name,
        'category'       : category,
        'plane_days'     : p,
        'freight_days'   : f,
        'ship_days'      : s,
        'reward_mean'    : round(float(r.mean()), 2),
        'reward_std'     : round(float(r.std()), 2),
        'steps_mean'     : round(float(s_a.mean()), 2),
        'complete_pct'   : round(float((s_a == 1000).mean() * 100), 2),
        'cv_ship'        : round(cv, 2),
        'skewness_ship'  : round(skew, 4),
        'reward_per_step': round(float(r.mean() / s_a.mean()), 4),
    }
 
 
# Main 
if __name__ == '__main__':
    for path in [MODEL_PATH, INVT_PATH, ORDER_PATH]:
        if not os.path.exists(path):
            print(f'EROARE: fisier negasit -> {path}')
            sys.exit(1)
 
    agent   = load_agent()
    results = []
 
    for i, (name, cat, p, f, s) in enumerate(SCENARIOS):
        print(f'[{i+1}/{len(SCENARIOS)}] {name:15s} P={p:>2} F={f:>2} S={s:>2} ...',
              end=' ', flush=True)
        res = run_scenario(name, cat, p, f, s, agent)
        results.append(res)
        print(f'reward={res["reward_mean"]:>8.1f}  complete={res["complete_pct"]:.1f}%')
 
    out_path = os.path.join(os.path.dirname(__file__), 'sensitivity_results.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
 
    print(f'\nGata! CSV salvat la: {out_path}')