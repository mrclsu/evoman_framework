from evoman.environment import Environment
import numpy

class NoTimeEnv(Environment):
    def __init__(self, *args, **kwargs):
        super(NoTimeEnv, self).__init__(*args, **kwargs)

    def fitness_single(self):
        return 0.9 * (100 - self.get_enemylife()) + 0.1 * self.get_playerlife() 


class TimeBonusEnv(Environment):
    def __init__(self, *args, **kwargs):
        super(TimeBonusEnv, self).__init__(*args, **kwargs)

    def fitness_single(self):
        time_bonus = numpy.log(self.get_time()) if self.get_enemylife() <= 0 else 0
        return 0.9 * (100 - self.get_enemylife()) + 0.1 * self.get_playerlife() + time_bonus 


def default_env(experiment_name, player_controller):
    return Environment(
            experiment_name=experiment_name,
            enemies=[2, 3, 4, 5, 6, 8],
            playermode="ai",
            player_controller=player_controller,
            enemymode="static",
            level=2,
            multiplemode="yes",
            speed="fastest",
            visuals=False
        )

def no_time_env(experiment_name, player_controller):
    return NoTimeEnv(
            experiment_name=experiment_name,
            enemies=[2, 3, 4, 5, 6, 8],
            playermode="ai",
            player_controller=player_controller,
            enemymode="static",
            level=2,
            multiplemode="yes",
            speed="fastest",
            visuals=False
        )

def time_bonus_env(experiment_name, player_controller):
    return TimeBonusEnv(
            experiment_name=experiment_name,
            enemies=[2, 3, 4, 5, 6, 8],
            playermode="ai",
            player_controller=player_controller,
            enemymode="static",
            level=2,
            multiplemode="yes",
            speed="fastest",
            visuals=False
        )