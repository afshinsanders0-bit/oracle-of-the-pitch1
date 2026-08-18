"""
Oracle Analyst — Generates Sharp Betting Previews
===================================================
Combines ML predictions with high-quality narrative analysis.
"""

import os
from typing import Dict, Any, Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. Install with: pip install python-dotenv")


class FootballAnalyst:
    """
    Oracle Analyst — Generates sharp, contextual football betting previews.
    Combines ML model predictions with high-quality narrative analysis.
    """

    def __init__(self, provider: str = "groq"):
        self.provider = provider.lower()
        if self.provider not in ("groq", "claude"):
            raise ValueError(f"Provider '{self.provider}' not supported. Use 'groq' or 'claude'.")
        self.system_prompt = self._build_system_prompt()
        self.client: Optional[Union[Any, Any]] = None
        print(f"✅ FootballAnalyst initialised with {self.provider.upper()}")

    def _build_system_prompt(self) -> str:
        return """You are a senior football betting analyst. You write institutional-grade match reports for professional punters. Your output is concise, factual, and directly actionable.

ABSOLUTE RULES — NO EXCEPTIONS
1. You MUST NOT invent player names, injury status, team stats, odds, or historical data.
2. You MUST ONLY use information provided in the user prompt.
3. If information is missing or unknown, write: "Not available" — do NOT guess or hallucinate.
4. You MUST ground every claim in the model predictions provided. If the model does not provide a stat, do not mention it.
5. You MUST NOT use outside knowledge about specific matches unless it is general football knowledge.
6. You MUST give realistic odds ranges based on typical bookmaker pricing for the market.
7. You MUST end with a FINAL VERDICT block. This is non-negotiable.
8. You MUST write in a confident, professional tone. No hedging. No filler.

6-SECTION STRUCTURE — use these exact headings in this exact order:
1. Match Header — 2-3 sentences max: form, motivation, stakes, key absences.
2. Detailed Context — injuries, rotation, travel, managerial changes, market movement.
3. Tactical — how styles clash, key matchups, likely in-game adjustments.
4. Supporting Trends — 3-4 bullets max with actual numbers from the model data.
5. Expected Outcome — likely result + the one main risk.

FINAL VERDICT — this is the most important section. End every report with this exact block:
FINAL VERDICT
Bet: [exact market selection, e.g. "Home Win" or "Over 2.5 Goals" or "BTTS"]
Odds: [realistic odds range you expect bookmakers to offer]
Stake: [units / % of bankroll]
Edge: [~X%]
Confidence: HIGH / MEDIUM / LOW
One-line reason: [why this bet has value]

If there is genuinely no value, write:
FINAL VERDICT
Bet: PASS
Odds: N/A
Stake: 0 units
Edge: N/A
Confidence: N/A
One-line reason: [why no value exists]

RULES
✓ Be specific. Name players ONLY if their status is provided in the data.
✓ Give realistic odds ranges. Typical ranges: Home Win 1.40-3.00, Draw 2.80-4.50, Away Win 1.80-5.00, Over 2.5 1.60-2.80, BTTS 1.60-2.20.
✓ Connect model numbers to narrative. If model says 62% Over 2.5, explain why using the data provided.
✓ If model confidence is LOW, say so plainly and reduce stake recommendation.
✓ Use precise language. Say "68% probability" not "likely". Say "edge of ~6%" not "good value".
✗ No filler. No "it's hard to predict." No fake confidence.
✗ Never invent statistics, injuries, or odds. If you don't have the data, say so."""

    def _get_client(self) -> Any:
        """Initialize and return the LLM client for the configured provider."""
        if self.client is not None:
            return self.client
        
        try:
            if self.provider == "groq":
                try:
                    from groq import Groq
                except ImportError:
                    raise ImportError(
                        "Groq SDK not installed. Install with: pip install groq"
                    )
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key or "your_groq_key" in api_key:
                    raise ValueError("GROQ_API_KEY is missing or invalid in .env file")
                self.client = Groq(api_key=api_key)
                print("✅ Groq client connected successfully!")
                return self.client
                
            elif self.provider == "claude":
                try:
                    import anthropic
                except ImportError:
                    raise ImportError(
                        "Anthropic SDK not installed. Install with: pip install anthropic"
                    )
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key or "your_anthropic_key" in api_key:
                    raise ValueError("ANTHROPIC_API_KEY is missing or invalid in .env file")
                self.client = anthropic.Anthropic(api_key=api_key)
                print("✅ Claude client connected successfully!")
                return self.client
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
        except Exception as e:
            print(f"❌ Failed to initialise {self.provider.upper()} client: {e}")
            raise

    @staticmethod
    def _extract_text_from_content(content_blocks: Any) -> str:
        """Extract text from Anthropic SDK response.content list."""
        if not content_blocks:
            return ""
        
        text_parts = []
        for block in content_blocks:
            if hasattr(block, "text") and isinstance(getattr(block, "text", None), str):
                text_parts.append(block.text)
        
        return "".join(text_parts).strip()

    def _get_available_groq_model(self, client) -> str:
        """Discover an available Groq model dynamically."""
        try:
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            
            text_models = [
                m for m in model_ids
                if not any(x in m.lower() for x in [
                    "whisper", "compound", "orpheus", "prompt-guard", "safeguard", "vision"
                ])
            ]
            
            if text_models:
                return text_models[0]
            
            if model_ids:
                return model_ids[0]
                
        except Exception:
            pass
        
        env_model = os.getenv("GROQ_MODEL")
        if env_model:
            return env_model
        
        raise ValueError(
            "Could not discover any Groq models. "
            "Set GROQ_MODEL env var to a valid model ID (e.g. llama-3.1-8b-instant)."
        )

    def generate_preview(
        self,
        match_data: Dict[str, Any],
        model_predictions: Dict[str, Any],
    ) -> str:
        """Generate a full betting preview for one match."""
        try:
            client = self._get_client()
            user_prompt = self._build_user_prompt(match_data, model_predictions)

            if self.provider == "groq":
                groq_model = self._get_available_groq_model(client)
                try:
                    response = client.chat.completions.create(
                        model=groq_model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.75,
                        max_tokens=2000,
                    )
                    result = response.choices[0].message.content.strip()
                except Exception as model_err:
                    if "model_not_found" in str(model_err) or "404" in str(model_err):
                        print(f"⚠️  Model {groq_model} unavailable, trying another...")
                        fallback_model = self._get_available_groq_model(client)
                        if fallback_model == groq_model:
                            fallback_model = None
                        if not fallback_model:
                            raise ValueError(
                                "No Groq text models are available for your account. "
                                "Check your Groq plan or set GROQ_MODEL env var to a valid model ID."
                            )
                        try:
                            response = client.chat.completions.create(
                                model=fallback_model,
                                messages=[
                                    {"role": "system", "content": self.system_prompt},
                                    {"role": "user", "content": user_prompt},
                                ],
                                temperature=0.75,
                                max_tokens=2000,
                            )
                            result = response.choices[0].message.content.strip()
                        except Exception:
                            raise ValueError(
                                f"Groq model {fallback_model} also unavailable. "
                                "Set GROQ_MODEL env var to a valid model ID from your Groq console."
                            )
                    else:
                        raise
                
            elif self.provider == "claude":
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=2000,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt},
                    ],
                )
                result = self._extract_text_from_content(response.content)
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            if not result:
                raise ValueError("Empty response from LLM")

            print("✅ Analyst preview generated successfully")
            return result

        except Exception as e:
            error_msg = f"❌ Analyst Error: {str(e)}"
            print(error_msg)
            return error_msg

    def _build_user_prompt(
        self,
        match_data: Dict[str, Any],
        model_pred: Dict[str, Any],
    ) -> str:
        """Build rich user prompt from match data and model predictions."""
        
        pred_lines = []
        key_labels = {
            "home_win_prob":        "Home Win probability",
            "draw_prob":            "Draw probability",
            "away_win_prob":        "Away Win probability",
            "btts_prob":            "BTTS (Both Teams to Score) probability",
            "over_25_prob":         "Over 2.5 Goals probability",
            "over_15_prob":         "Over 1.5 Goals probability",
            "over_35_prob":         "Over 3.5 Goals probability",
            "corners_over_prob":    "Corners Over 9.5 probability",
            "expected_home_goals":  "Expected home goals (xG proxy)",
            "expected_away_goals":  "Expected away goals (xG proxy)",
            "expected_total_goals": "Expected total goals",
            "home_elo":             "Home team ELO rating",
            "away_elo":             "Away team ELO rating",
            "elo_diff":             "ELO difference (home minus away)",
            "home_form":            "Home team recent form score (0-1)",
            "away_form":            "Away team recent form score (0-1)",
            "h2h_avg_goals":        "H2H average goals per game",
            "h2h_btts_rate":        "H2H BTTS rate",
            "confidence":           "Model overall confidence level",
            "top_outcome":          "Model top pick",
            "top_probability":      "Model top pick probability",
        }

        for key, label in key_labels.items():
            if key in model_pred:
                val = model_pred[key]
                if "prob" in key or "rate" in key or "form" in key:
                    try:
                        pred_lines.append(f"  • {label}: {float(val):.1%}")
                    except (ValueError, TypeError):
                        pred_lines.append(f"  • {label}: {val}")
                elif isinstance(val, float):
                    pred_lines.append(f"  • {label}: {val:.3f}")
                else:
                    pred_lines.append(f"  • {label}: {val}")

        for key, val in model_pred.items():
            if key not in key_labels:
                pred_lines.append(f"  • {key}: {val}")

        pred_summary = "\n".join(pred_lines) if pred_lines else "No model predictions available."

        confidence_warning = ""
        if model_pred.get("confidence") == "LOW":
            confidence_warning = "\n⚠️  WARNING: Model confidence is LOW. Predictions may be unreliable. Reduce stake size accordingly."
        
        missing = [label for key, label in key_labels.items() if key not in model_pred or model_pred.get(key) is None]
        if missing and model_pred.get("confidence") != "LOW":
            confidence_warning = f"\n⚠️  NOTE: {len(missing)} model outputs are missing: {', '.join(missing[:5])}. Analysis should rely more heavily on available data."

        analyst_notes = match_data.get("analyst_notes", "")
        notes_section = ""
        if analyst_notes:
            notes_section = f"\nUSER-PROVIDED CONTEXT (treat as factual):\n{analyst_notes}\n"

        return f"""
══════════════════════════════════════════════════════════════════════════════
MATCH TO ANALYSE
══════════════════════════════════════════════════════════════════════════════

Match    : {match_data.get('home_team')} vs {match_data.get('away_team')}
League   : {match_data.get('league')}
Date/Time: {match_data.get('date', 'Upcoming')}
Status   : {match_data.get('status', 'Unknown')}
Matchday : {match_data.get('matchday', 'Unknown')}

══════════════════════════════════════════════════════════════════════════════
MODEL PREDICTIONS — TREAT THESE AS YOUR SOURCE OF TRUTH
══════════════════════════════════════════════════════════════════════════════
{pred_summary}
{confidence_warning}
{notes_section}

══════════════════════════════════════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════════════════════════════════════

Write a sharp, professional betting preview for this match. Follow these rules:

1. STRUCTURE — use exactly these 6 sections in this order:
   **Match Header**
   **Detailed Context**
   **Tactical**
   **Supporting Trends**
   **Expected Outcome**
   **FINAL VERDICT**

2. TONE — confident, precise, professional. No hedging. No filler. Use exact numbers from the model data.

3. ACCURACY — ONLY use the model predictions and analyst notes above. If a stat is missing, write "Not available." Never invent data.

4. BETTING FOCUS — every section should lead toward the bet. The FINAL VERDICT is the climax of the report.

5. FINAL VERDICT FORMAT — end with this exact block:
   FINAL VERDICT
   Bet: [exact market]
   Odds: [realistic range]
   Stake: [units / %]
   Edge: [~X%]
   Confidence: HIGH / MEDIUM / LOW
   One-line reason: [why this bet has value]

   If no value exists, write PASS with a reason.

6. EDGE CALCULATION — calculate edge as: (model_probability × odds) - 1. Express as percentage. Be realistic.

7. STAKE GUIDANCE — HIGH confidence = 2.0-3.0 units, MEDIUM = 1.0-1.5 units, LOW = 0.5 units or PASS.

Do NOT write a generic preview. This is a professional betting report."""
