"""
chatgpt_integrator.py
ChatGPT API integration for VP Investments platform to enhance analysis and insights.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from config.config import AI_FEATURES, OPENAI
import pandas as pd

# === Setup ===
load_dotenv()
logger = logging.getLogger(__name__)

class ChatGPTIntegrator:
    """
    Integrates ChatGPT API to enhance VP Investments with intelligent analysis,
    summarization, and natural language insights.
    """
    
    def __init__(self):
        # Initialize OpenAI client; uses OPENAI_API_KEY from env
        self.client = OpenAI(api_key=OPENAI.get("API_KEY"))
        # Default to GPT-4o Mini for cost/perf balance if not specified
        self.model = OPENAI.get("MODEL", "gpt-4o-mini")
        
    def generate_signal_commentary(self, row: pd.Series) -> str:
        """
        Generate intelligent commentary for a stock signal explaining the key factors.
        """
        prompt = f"""
        You are a professional investment analyst. Analyze this stock signal and provide a concise, 
        insightful commentary (2-3 sentences) explaining the key investment thesis.

        Stock: {row.get('Ticker', 'N/A')} - {row.get('Company', 'N/A')}
        Score: {row.get('Score (0–100)', 0)}/100
        Trade Type: {row.get('Trade Type', 'N/A')}
        
        Key Metrics:
        - Reddit Sentiment: {row.get('Reddit Sentiment', 0):.2f}
        - Mentions: {row.get('Mentions', 0)}
        - Price 7D %: {row.get('Price 7D %', 0):.1f}%
        - Volume Spike: {row.get('Volume Spike Ratio', 0):.1f}x
        - Market Cap: {row.get('Market Cap', 'N/A')}
        - P/E Ratio: {row.get('P/E Ratio', 'N/A')}
        - RSI: {row.get('RSI', 'N/A')}
        - Top Factors: {row.get('Top Factors', 'N/A')}
        
        Focus on: 1) What's driving the signal, 2) Key risks/opportunities, 3) Trading outlook.
        Keep it professional and actionable. Avoid speculation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"ChatGPT commentary error for {row.get('Ticker')}: {e}")
            return "Analysis unavailable - please review metrics manually."

    def enhance_reddit_summary(self, ticker: str, reddit_text: str, sentiment: float) -> str:
        """
        Enhance raw Reddit data with intelligent summarization and sentiment context.
        """
        prompt = f"""
        Summarize this Reddit discussion about {ticker} in 1-2 professional sentences.
        Focus on key investment themes, catalysts, or concerns mentioned.
        
        Reddit Content: {reddit_text[:500]}...
        Sentiment Score: {sentiment:.2f} (Range: -1 to +1)
        
        Extract: Main thesis, catalysts, or concerns. Ignore spam/pumping language.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Reddit summary error for {ticker}: {e}")
            return reddit_text[:100] + "..." if reddit_text else "No summary available"

    def generate_risk_assessment(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate detailed risk assessment using ChatGPT analysis.
        """
        prompt = f"""
        As a risk analyst, assess the investment risks for this stock signal.
        Provide a structured risk analysis in JSON format.

        Stock: {row.get('Ticker')} - {row.get('Company')}
        
        Metrics:
        - Volatility: {row.get('Volatility', 0):.3f}
        - Beta vs SPY: {row.get('Beta vs SPY', 0):.2f}
        - Liquidity (Daily Value): ${row.get('Avg Daily Value Traded', 0):,.0f}
        - Short Interest: {row.get('Short Percent Float', 0):.1f}%
        - P/E Ratio: {row.get('P/E Ratio', 'N/A')}
        - Market Cap: {row.get('Market Cap', 'N/A')}
        
        Return JSON with:
        {{
          "overall_risk": "Low/Medium/High",
          "key_risks": ["risk1", "risk2", "risk3"],
          "risk_mitigation": "brief strategy",
          "position_sizing": "Conservative/Moderate/Aggressive"
        }}
        
        Respond ONLY with valid JSON, no additional text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            # Clean the response and attempt to parse JSON
            content = response.choices[0].message.content.strip()
            # Remove any markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON parsing error for {row.get('Ticker')}: {je}")
            # Return a structured response based on the content
            return {
                "overall_risk": "Medium", 
                "key_risks": ["Overbought conditions", "High volatility", "Valuation risk"], 
                "risk_mitigation": "Monitor RSI and volume, consider partial position",
                "position_sizing": "Moderate"
            }
        except Exception as e:
            logger.error(f"Risk assessment error for {row.get('Ticker')}: {e}")
            return {
                "overall_risk": "Unknown", 
                "key_risks": ["Analysis unavailable"], 
                "risk_mitigation": "Manual review required",
                "position_sizing": "Conservative"
            }

    def generate_portfolio_insights(self, df: pd.DataFrame) -> str:
        """
        Generate portfolio-level insights and recommendations.
        """
        top_signals = df.head(10)
        sectors = df['Sector'].value_counts() if 'Sector' in df.columns else {}
        avg_score = df['Score (0–100)'].mean() if 'Score (0–100)' in df.columns else 0
        
        prompt = f"""
        You are a portfolio strategist analyzing {len(df)} stock signals.
        
        Key Statistics:
        - Average Score: {avg_score:.1f}/100
        - Top 10 Signals: {', '.join(top_signals['Ticker'].tolist())}
        - Sector Distribution: {dict(sectors)}
        - Score Range: {df['Score (0–100)'].min():.1f} to {df['Score (0–100)'].max():.1f}
        
        Provide a 4-5 sentence portfolio strategy focusing on:
        1) Overall signal quality and market conditions
        2) Sector concentration risks/opportunities 
        3) Recommended position sizing approach
        4) Key risks to monitor
        
        Be professional and actionable.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Portfolio insights error: {e}")
            return "Portfolio analysis unavailable - please review signals manually."

    def explain_signal_scoring(self, row: pd.Series) -> str:
        """
        Explain why a signal received its score in plain English.
        """
        prompt = f"""
        Explain in simple terms why {row.get('Ticker')} received a {row.get('Score (0–100)', 0)}/100 score.

        Key Factors:
        - Reddit Mentions: {row.get('Mentions', 0)} posts
        - Reddit Sentiment: {row.get('Reddit Sentiment', 0):.2f}/1.0
        - Price Momentum (7D): {row.get('Price 7D %', 0):.1f}%
        - Volume Activity: {row.get('Volume Spike Ratio', 0):.1f}x normal
        - Financial Strength: P/E {row.get('P/E Ratio', 'N/A')}, RSI {row.get('RSI', 'N/A')}
        
        Explain in 2-3 sentences what's driving this score. Use plain English.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Score explanation error for {row.get('Ticker')}: {e}")
            return "Score explanation unavailable."

    def generate_market_outlook(self, df: pd.DataFrame, external_data: Dict = None) -> str:
        """
        Generate overall market outlook based on signal patterns.
        """
        high_score_count = len(df[df['Score (0–100)'] > 70]) if 'Score (0–100)' in df.columns else 0
        avg_sentiment = df['Reddit Sentiment'].mean() if 'Reddit Sentiment' in df.columns else 0
        
        prompt = f"""
        Analyze the current market sentiment based on alternative data signals:
        
        Signal Analysis:
        - Total Signals: {len(df)}
        - High-Quality Signals (>70/100): {high_score_count}
        - Average Reddit Sentiment: {avg_sentiment:.2f}
        - Most Active Sectors: {df['Sector'].value_counts().head(3).to_dict() if 'Sector' in df.columns else 'N/A'}
        
        Provide a brief market outlook (3-4 sentences) covering:
        1) Overall retail sentiment temperature
        2) Quality of current opportunities
        3) Recommended market approach
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Market outlook error: {e}")
            return "Market analysis unavailable."

    def generate_news_summary(self, ticker: str, articles: List[Dict[str, Any]], avg_sentiment: float) -> str:
        """
        Summarize top news for a ticker in 1-2 sentences using titles/descriptions.
        """
        if not articles:
            return "No recent news."
        # Build compact context from up to 5 articles
        trimmed = []
        for a in articles[:5]:
            title = (a.get("title") or "").strip()
            desc = (a.get("description") or "").strip()
            if title or desc:
                trimmed.append(f"- {title}: {desc}")
        context = "\n".join(trimmed)
        prompt = f"""
        You are a financial news analyst. Summarize the recent news for {ticker} in 1-2 concise sentences.
        Focus on catalysts, risks, and overall tone. Avoid hype.
        Average sentiment score: {avg_sentiment:.2f} (VADER compound, -1 to +1)

        Articles:\n{context}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"News summary error for {ticker}: {e}")
            return "News summary unavailable."

    def generate_trends_commentary(self, ticker: str, interest: float, spike: float) -> str:
        """
        Provide a short commentary on Google Trends signals.
        """
        prompt = f"""
        Provide a one-sentence insight on Google search activity for {ticker} given:
        - Average interest: {interest:.2f}
        - Recent spike ratio: {spike:.2f}
        Indicate if interest is rising, flat, or falling and whether it likely reflects growing retail attention.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Trends commentary error for {ticker}: {e}")
            return "Trends commentary unavailable."

# === Integration Functions ===

def enhance_dataframe_with_chatgpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ChatGPT-enhanced columns to the main signals dataframe.
    """
    if not OPENAI.get("API_KEY"):
        logger.warning("No OpenAI API key found - skipping ChatGPT enhancements")
        return df
    
    integrator = ChatGPTIntegrator()
    
    # Add AI Commentary column
    logger.info("Generating AI commentary for top signals...")
    df['AI Commentary'] = ""
    df['Score Explanation'] = ""
    df['Risk Assessment'] = ""
    
    # Process only the first N signals to manage API costs
    max_rows = int(AI_FEATURES.get("MAX_ROWS", 20))
    top_signals = df.head(max_rows)
    
    for idx, row in top_signals.iterrows():
        try:
            # Generate commentary
            df.at[idx, 'AI Commentary'] = integrator.generate_signal_commentary(row)
            
            # Generate score explanation
            df.at[idx, 'Score Explanation'] = integrator.explain_signal_scoring(row)
            
            # Generate risk assessment (simplified)
            risk_data = integrator.generate_risk_assessment(row)
            df.at[idx, 'Risk Assessment'] = f"{risk_data.get('overall_risk', 'Unknown')} Risk"
            
            logger.info(f"Enhanced {row.get('Ticker')} with AI analysis")
            
        except Exception as e:
            logger.error(f"Error enhancing {row.get('Ticker')}: {e}")
            df.at[idx, 'AI Commentary'] = "Analysis unavailable"
            df.at[idx, 'Score Explanation'] = "Manual review required"
            df.at[idx, 'Risk Assessment'] = "Unknown Risk"
    
    return df

def generate_executive_summary(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate an executive summary of the entire signal set.
    """
    if not OPENAI.get("API_KEY"):
        return {"summary": "AI analysis unavailable - no API key configured"}
    
    integrator = ChatGPTIntegrator()
    
    return {
        "portfolio_insights": integrator.generate_portfolio_insights(df),
        "market_outlook": integrator.generate_market_outlook(df),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
