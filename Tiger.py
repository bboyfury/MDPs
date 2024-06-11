
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy


class TigerState(pomdp_py.State):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, TigerState):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "TigerState(%s)" % self.name

    def other(self):
        if self.name.endswith("left"):
            return TigerState("tiger-right")
        else:
            return TigerState("tiger-left")


class TigerAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, TigerAction):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "TigerAction(%s)" % self.name


class TigerObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, TigerObservation):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "TigerObservation(%s)" % self.name


# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            # heard the correct growl
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0, 1) < thresh:
            return TigerObservation(next_state.name)
        else:
            return TigerObservation(next_state.other().name)

    def get_all_observations(self):

        return [TigerObservation(s) for s in {"tiger-left", "tiger-right"}]


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):

        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return TigerState(state.name)

    def get_all_states(self):

        return [TigerState(s) for s in {"tiger-left", "tiger-right"}]


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else:  # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):


    ACTIONS = [TigerAction(s) for s in {"open-left", "open-right", "listen"}]

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


class TigerProblem(pomdp_py.POMDP):

    def __init__(self, obs_noise, init_true_state, init_belief):
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(),
            ObservationModel(obs_noise),
            RewardModel(),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(), RewardModel())
        super().__init__(agent, env, name="TigerProblem")

    @staticmethod
    def create(state="tiger-left", belief=0.5, obs_noise=0.15):

        init_true_state = TigerState(state)
        init_belief = pomdp_py.Histogram(
            {TigerState("tiger-left"): belief, TigerState("tiger-right"): 1.0 - belief}
        )
        tiger_problem = TigerProblem(obs_noise, init_true_state, init_belief)
        tiger_problem.agent.set_belief(init_belief, prior=True)
        return tiger_problem


def test_planner(tiger_problem, planner, nsteps=3, debug_tree=False):

    for i in range(nsteps):
        action = planner.plan(tiger_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {tiger_problem.env.state}")
        print(f"Belief: {tiger_problem.agent.cur_belief}")
        print(f"Action: {action}")

        reward = tiger_problem.env.reward_model.sample(
            tiger_problem.env.state, action, None
        )
        print("Reward:", reward)

        real_observation = TigerObservation(tiger_problem.env.state.name)
        print(">> Observation:", real_observation)
        tiger_problem.agent.update_history(action, real_observation)

        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                tiger_problem.agent.cur_belief,
                action,
                real_observation,
                tiger_problem.agent.observation_model,
                tiger_problem.agent.transition_model,
            )
            tiger_problem.agent.set_belief(new_belief)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken
            # until every time door is opened.
            print("\n")


def make_tiger(noise=0.15, init_state="tiger-left", init_belief=[0.5, 0.5]):

    tiger = TigerProblem(
        noise,
        TigerState(init_state),
        pomdp_py.Histogram(
            {
                TigerState("tiger-left"): init_belief[0],
                TigerState("tiger-right"): init_belief[1],
            }
        ),
    )
    return tiger


def main():
    init_true_state = random.choice(["tiger-left", "tiger-right"])
    init_belief = pomdp_py.Histogram(
        {TigerState("tiger-left"): 0.5, TigerState("tiger-right"): 0.5}
    )
    tiger = make_tiger(init_state=init_true_state)

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(tiger, vi, nsteps=8)

    


if __name__ == "__main__":
    main()