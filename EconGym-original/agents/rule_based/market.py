import numpy as np


class MarketRules:
    """
    Rule set for market/firm actions.
    """
    
    @staticmethod
    def get_action(type,obs, action_dim):
        """
        Generate actions for the firms based on the market type and observations.
        Each market type will have different rules for setting prices and wage rates.
        
        Many economic methods require full knowledge of the environment's transition,
        including the consumer demand function, which must be learned from trajectory data in the field of AI.
        No explicit economic rules are provided; please design your own for testing.
        """
        # Retrieve the number of firms from the observation
        firm_n = len(obs)
        
        # Handle different market types
        if type == "perfect":
            return np.random.randn(firm_n, action_dim)   # Firms in perfect competition have no pricing power.
        
        elif type == "monopoly":
            return MarketRules._monopoly(firm_n, action_dim, obs)
        
        elif type == "oligopoly":
            return MarketRules._oligopoly(firm_n, action_dim, obs)
        
        elif type == "monopolistic_competition":
            return MarketRules._monopolistic_competition(firm_n, action_dim, obs)
        else:
            raise ValueError("Unsupported market type.")
    

    
    @staticmethod
    def _monopoly(firm_n, action_dim, obs):
        """
        Market type: Monopoly.
        A single firm controls the market. It maximizes profit by adjusting price and wage rate.
        """
        firm_productivity = obs[:, 1].item()
        
        # Monopoly will set prices high to maximize profit, and adjust wages based on productivity.
        price = np.full((firm_n, 1),firm_productivity * 2)  # Set price based on productivity (doubled for profit margin)
        wage_rate = np.full((firm_n, 1), firm_productivity * 0.5)  # Wages set as a fraction of productivity
        return np.hstack([price, wage_rate])
    
    @staticmethod
    def _oligopoly(firm_n, action_dim, obs):
        """
        Market type: Oligopoly.
        A few firms dominate the market. They set prices and wages considering competition.
        """
        firm_productivity = obs[:, 1]
        
        # Oligopoly firms set prices considering competition but also maximize profit
        price = firm_productivity.reshape(-1, 1) * np.random.uniform(1.2, 1.5, (firm_n, 1))  # Price is influenced by productivity and competition
        wage_rate = firm_productivity.reshape(-1, 1) * np.random.uniform(0.6, 0.9, (firm_n, 1))  # Wage rate depends on competition level
        return np.hstack([price, wage_rate])
    
    @staticmethod
    def _monopolistic_competition(firm_n, action_dim, obs):
        """
        Market type: Monopolistic competition.
        Many firms exist, each offering differentiated products. Firms adjust prices based on their market position.
        """
        firm_productivity = obs[:, 1]
        
        # In monopolistic competition, firms set prices based on their product differentiation.
        price = firm_productivity.reshape(-1, 1) * np.random.uniform(1.1, 1.3, (firm_n, 1))  # Price slightly above the market average
        wage_rate = firm_productivity.reshape(-1, 1) * np.random.uniform(0.7, 1.0, (firm_n, 1))  # Wages based on product differentiation
        return np.hstack([price, wage_rate])
    

