"""
FASE 6: DEFI & AUTOMATED MARKET MAKER (AMM) QUANTITATIVE ANALYTICS
1. Uniswap V2 vs V3 Concentrated Liquidity Mathematics
2. Impermanent Loss (IL) Exact Modeling & Capital Efficiency Multiplier
3. Optimal LP Tick Range Optimization as a function of Daily Volatility
4. MEV Sandwich Attack Slippage Extraction Model
"""

import numpy as np
import scipy.stats as stats

# ─────────────────────────────────────────────────────────────
# 1. IMPERMANENT LOSS MATHEMATICS
# ─────────────────────────────────────────────────────────────

def impermanent_loss_v2(price_ratio):
    """
    Standard Constant Product AMM (Uniswap V2) Impermanent Loss:
    IL(k) = (2 * sqrt(k)) / (1 + k) - 1, where k = P_t / P_0
    """
    k = price_ratio
    return (2.0 * np.sqrt(k)) / (1.0 + k) - 1.0

def impermanent_loss_v3_concentrated(p0, pt, pa, pb):
    """
    Uniswap V3 Concentrated Liquidity Impermanent Loss within range [pa, pb].
    When price stays within [pa, pb], capital efficiency increases by:
    Multiplier = 1 / (1 - sqrt(pa / pb))
    """
    if pt <= pa:
        # 100% token Y, 0% token X
        v_hold = 0.5 * (1.0 + pt / p0)
        v_lp = np.sqrt(pa * pb) / (np.sqrt(pb) - np.sqrt(pa)) * (pt / pa - 1.0) # simplified normalized
    elif pt >= pb:
        # 100% token X, 0% token Y
        pass

    # Standard formula for V3 relative to hold
    sqrt_p0 = np.sqrt(p0)
    sqrt_pt = np.sqrt(pt)
    sqrt_pa = np.sqrt(pa)
    sqrt_pb = np.sqrt(pb)
    
    # Capital efficiency multiplier vs full range
    cap_eff = 1.0 / (1.0 - np.sqrt(pa / pb))
    il_v2 = impermanent_loss_v2(pt / p0)
    il_v3 = il_v2 * cap_eff  # First order approximation within range
    return il_v2, il_v3, cap_eff

# ─────────────────────────────────────────────────────────────
# 2. OPTIMAL LP RANGE AS A FUNCTION OF VOLATILITY
# ─────────────────────────────────────────────────────────────

def optimal_lp_range(spot_price=80000.0, ann_vol=0.5261, holding_days=30, confidence=0.95):
    """
    Calculates optimal tick boundary [pa, pb] for concentrated liquidity:
    Range = spot * exp( +/- z_alpha * sigma * sqrt(dt) )
    Balances fee accrual time vs out-of-range divergence risk.
    """
    dt = holding_days / 365.0
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    sigma_period = ann_vol * np.sqrt(dt)
    
    pa = spot_price * np.exp(-z * sigma_period)
    pb = spot_price * np.exp(+z * sigma_period)
    cap_efficiency = 1.0 / (1.0 - np.sqrt(pa / pb))
    
    return {
        "spot": spot_price,
        "lower_pa": pa,
        "upper_pb": pb,
        "bandwidth_pct": (pb - pa) / spot_price * 100.0,
        "capital_efficiency_multiplier": cap_efficiency,
        "prob_in_range": confidence * 100.0
    }

# ─────────────────────────────────────────────────────────────
# 3. MEV SANDWICH ATTACK SLIPPAGE & ARBITRAGE SHORTFALL
# ─────────────────────────────────────────────────────────────

def simulate_sandwich_attack(
    pool_reserve_x=100.0, pool_reserve_y=8000000.0, # 100 BTC, $8M USDC pool
    victim_swap_usd=200000.0, slippage_tolerance=0.01
):
    """
    Models the economic extraction of a front-running/back-running sandwich bot:
    1. Front-run: Bot buys asset ahead of victim, pushing price up to slippage limit.
    2. Victim swap executes at the worst possible execution price.
    3. Back-run: Bot sells back to pool, harvesting victim's slippage.
    """
    # Initial constant product k = x * y
    k = pool_reserve_x * pool_reserve_y
    initial_price = pool_reserve_y / pool_reserve_x
    max_victim_price = initial_price * (1.0 + slippage_tolerance)
    
    # How much USDC can bot inject to push price exactly to max_victim_price?
    # New reserves: y_new / x_new = max_victim_price => y_new^2 / k = max_price
    y_target = np.sqrt(k * max_victim_price)
    bot_in_usdc = y_target - pool_reserve_y
    
    if bot_in_usdc <= 0:
        return {"extractable_profit_usd": 0.0, "victim_loss_usd": 0.0}
        
    # Pool state after front-run:
    x_after_bot = k / y_target
    bot_received_btc = pool_reserve_x - x_after_bot
    
    # Victim executes: injects victim_swap_usd
    y_after_victim = y_target + victim_swap_usd
    x_after_victim = k / y_after_victim
    victim_received_btc = x_after_bot - x_after_victim
    victim_effective_price = victim_swap_usd / victim_received_btc
    
    # Bot back-runs: sells bot_received_btc back
    x_final = x_after_victim + bot_received_btc
    y_final = k / x_final
    bot_payout_usdc = y_after_victim - y_final
    bot_profit = bot_payout_usdc - bot_in_usdc
    
    # Fair execution without sandwich
    fair_x = k / (pool_reserve_y + victim_swap_usd)
    fair_btc = pool_reserve_x - fair_x
    victim_shortfall_usd = (fair_btc - victim_received_btc) * initial_price
    
    return {
        "bot_capital_required_usd": bot_in_usdc,
        "bot_extracted_profit_usd": bot_profit,
        "victim_shortfall_usd": victim_shortfall_usd,
        "victim_execution_price": victim_effective_price,
        "price_impact_pct": ((victim_effective_price / initial_price) - 1.0) * 100.0
    }

if __name__ == "__main__":
    print("=" * 60)
    print("FASE 6: DEFI & AMM QUANTITATIVE ANALYTICS")
    print("=" * 60)

    # 1. Impermanent Loss Profile
    print("\n[1] Impermanent Loss (IL) Profile vs Spot Price Movement:")
    price_moves = [0.50, 0.75, 0.90, 1.10, 1.25, 1.50, 2.00]
    print("  Price Change | V2 AMM Loss | Concentrated V3 Loss (5x Eff)")
    print("  -------------+-------------+------------------------------")
    for pm in price_moves:
        il_v2 = impermanent_loss_v2(pm)
        il_v3 = il_v2 * 5.0 # Assuming 5x capital concentration
        print(f"     {pm*100:5.1f}%    |   {il_v2*100:6.2f}%   |          {il_v3*100:6.2f}%")

    # 2. Optimal LP Range Optimization
    print("\n[2] Optimal Concentrated LP Range Optimization (30-Day Tenor, 95% Conf):")
    lp_opt = optimal_lp_range(spot_price=80000.0, ann_vol=0.5261, holding_days=30, confidence=0.95)
    print(f"  Current Spot Price:           ${lp_opt['spot']:,.2f}")
    print(f"  Optimal Lower Bound (pa):     ${lp_opt['lower_pa']:,.2f}")
    print(f"  Optimal Upper Bound (pb):     ${lp_opt['upper_pb']:,.2f}")
    print(f"  Total Bandwidth:              {lp_opt['bandwidth_pct']:.1f}%")
    print(f"  Capital Efficiency Multiplier:{lp_opt['capital_efficiency_multiplier']:.2f}x vs Uniswap V2")
    print(f"  Probability Staying in Range: {lp_opt['prob_in_range']:.1f}%")

    # 3. MEV Sandwich Attack Simulation
    print("\n[3] MEV Sandwich Attack Mechanics (DEX Swap $200k, 1% Max Slippage):")
    mev = simulate_sandwich_attack(pool_reserve_x=100.0, pool_reserve_y=8000000.0, victim_swap_usd=200000.0, slippage_tolerance=0.01)
    print(f"  Front-Running Capital Injected: ${mev['bot_capital_required_usd']:,.2f}")
    print(f"  Bot Net Extracted MEV Profit:  ${mev['bot_extracted_profit_usd']:,.2f}")
    print(f"  Victim Execution Shortfall:     ${mev['victim_shortfall_usd']:,.2f}")
    print(f"  Victim Execution Price:         ${mev['victim_execution_price']:,.2f} (+{mev['price_impact_pct']:.2f}% slippage)")
    print("=" * 60)
