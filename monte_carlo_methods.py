from .TicTacToe import TicTacToe
from .monte_carlo_ess import monte_carlo_es
from .off_monte_carlo_control import OFF_policy_monte_carlo
from .on_policy_first_visit_monte_carlo_control import on_policy_first_visit_monte_carlo_control
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2



def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:

    """
        Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
        Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
        Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
        """

    env = TicTacToe(3)
    return monte_carlo_es(env, 1.0, 0.9, 100)

def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """

    env = TicTacToe(3)
    return on_policy_first_visit_monte_carlo_control(env, 1.0, 0.9, 100)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe(3)
    return OFF_policy_monte_carlo(env, 1.0, 0.9, 100)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    return monte_carlo_es(env, 1.0, 0.9, 100)


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    return monte_carlo_es(env, 1.0, 0.9, 100)


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    return OFF_policy_monte_carlo(env, 1.0, 0.9, 100)


def demo():
    print(monte_carlo_es_on_tic_tac_toe_solo())
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print(monte_carlo_es_on_secret_env2())
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())
