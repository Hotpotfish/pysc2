from pysc2.agents.myAgent.myAgent_10.net.bicnet_for_level_2.bicnet_for_level_2_actor import bicnet_actor as  bicnet_actor
from pysc2.agents.myAgent.myAgent_10.net.bicnet_for_level_2.bicnet_for_level_2_cirtic import bicnet_critic as  bicnet_critic


class bicnet():

    def __init__(self, mu, sigma, learning_rate, action_dim, parameterdim, statedim, agents_number, enemy_number, name):
        actor = bicnet_actor(mu, sigma, learning_rate, action_dim, parameterdim, statedim, agents_number, enemy_number, name)

        critic = bicnet_critic(mu, sigma, learning_rate, action_dim, parameterdim, statedim, agents_number, enemy_number, name)
