import logging
import json
import asyncio
import aiohttp
from typing import TypedDict
from rag_service import RAGService

class LangGraphAgent:
    """AI agent using Google's Gemini API to recommend a strategy from a full suite."""
    def __init__(self, config, rag_service: RAGService):
        self.config = config
        self.rag_service = rag_service
        self.api_key = config.get('google_api', {}).get('api_key', "")
        self.model_name = "gemini-2.0-flash"

    async def get_recommended_strategy(self, market_conditions: set, user_prompt: str = None, rag_context: str = None):
        """
        Gets a strategy recommendation from the Gemini API, optionally augmented with
        RAG context.
        """
        if not self.api_key:
            logging.error("[Gemini Agent] Google API key not found. Defaulting strategy.")
            return "Gemini_Default"
            
        logging.info(f"[Gemini Agent] Market Conditions: {market_conditions}. Recommending strategy...")

        prompt_sections = [
            "You are an expert intraday options trading strategist for the Indian NIFTY 50 index.",
            "Your task is to select the single best strategy for today based on the provided data.",
            f"\n**Today's Market Conditions:** {', '.join(market_conditions)}",
        ]
        
        # --- FIX: Conditionally add the RAG context to the prompt ---
        if rag_context:
            logging.info("[Gemini Agent] Using RAG context for strategy selection.")
            prompt_sections.append(f"\n**RAG Context (Historical Performance):**\n{rag_context}")
        else:
            logging.info("[Gemini Agent] Bypassing RAG context for strategy selection.")

        if user_prompt:
            prompt_sections.append(f"\n**User's Preference/Observation:** '{user_prompt}'")

        prompt_sections.append("\n**Available Strategies (and their primary purpose):**")
        prompt_sections.append(
            """
1.  **'Gemini_Default'**: A balanced, multi-indicator strategy (CPR, EMA, RSI Divergence).
2.  **'Supertrend_MACD'**: A strong trend-following strategy.
3.  **'Volatility_Cluster_Reversal'**: A counter-trend strategy for high volatility.
4.  **'Volume_Spread_Analysis'**: Detects smart money activity.
5.  **'EMA_Cross_RSI'**: A classic, fast-acting momentum strategy.
6.  **'Momentum_VWAP_RSI'**: A momentum strategy using VWAP.
7.  **'Breakout_Prev_Day_HL'**: A breakout strategy on previous day's high/low.
8.  **'Opening_Range_Breakout'**: A classic ORB strategy.
9.  **'BB_Squeeze_Breakout'**: A volatility breakout strategy.
10. **'MA_Crossover'**: A simple moving average crossover strategy.
11. **'RSI_Divergence'**: A pure reversal strategy on RSI divergence.
12. **'Reversal_Detector'**: A specialized reversal strategy for overextended trends.
"""
        )
        prompt_sections.append("\nBased on all the above information, which single strategy name from the list has the highest probability of success today? Return only the name.")
        
        prompt = "\n".join(prompt_sections)
        
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()

            recommended_strategy = result["candidates"][0]["content"]["parts"][0]["text"].strip().replace("'", "").split('\n')[-1]

            valid_strategies = [
                "Gemini_Default", "Supertrend_MACD", "Volatility_Cluster_Reversal", 
                "Volume_Spread_Analysis", "EMA_Cross_RSI", "Momentum_VWAP_RSI",
                "Breakout_Prev_Day_HL", "Opening_Range_Breakout", "BB_Squeeze_Breakout",
                "MA_Crossover", "RSI_Divergence", "Reversal_Detector"
            ]
            if recommended_strategy not in valid_strategies:
                logging.warning(f"[Gemini Agent] LLM recommended unknown strategy: '{recommended_strategy}'. Defaulting.")
                recommended_strategy = "Gemini_Default"
            
            logging.info(f"[Gemini Agent] AI Recommended Strategy: {recommended_strategy}")
            return recommended_strategy

        except Exception as e:
            logging.error(f"[Gemini Agent] Error calling Gemini API: {e}. Defaulting to Gemini_Default.")
            return "Gemini_Default"
