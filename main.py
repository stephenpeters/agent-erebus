#!/usr/bin/env python3
"""
Erebus - Authenticity & Post-Processing Agent

Removes detectable AI fingerprints and reintroduces human irregularity.

Techniques:
- Clause deformation (subtle restructuring)
- Rhythm irregularity (breaking uniform patterns)
- Entropy injection (natural linguistic variation)
- Pattern disruption (removing AI markers)
"""

import os
import re
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from anthropic import Anthropic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Erebus - Authenticity & Post-Processing Agent",
    description="Removes AI fingerprints and reintroduces human irregularity",
    version="1.0.0"
)

# Initialize Anthropic
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Data directory
data_dir_env = os.getenv("MNEMOSYNE_DATA_DIR")
if data_dir_env:
    DATA_DIR = Path(data_dir_env).expanduser()
else:
    DATA_DIR = Path.home() / ".mnemosyne"

VOICEPRINT_PATH = DATA_DIR / "voiceprint.json"
CLEANED_DIR = DATA_DIR / "cleaned"

# Ensure directories exist
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Models
# ============================================================================

class DraftInput(BaseModel):
    """Input draft from IRIS"""
    draft_id: Optional[str] = None
    title: str
    content: str
    word_count: int
    metadata: Optional[Dict] = {}

class CleanedResponse(BaseModel):
    """Cleaned content response"""
    cleaned_id: str
    draft_id: Optional[str] = None
    title: str
    original_content: str
    cleaned_content: str
    diff_summary: str
    ai_likelihood_before: float
    ai_likelihood_after: float
    voice_deviation: float
    edit_distance: float
    word_count_change: int
    created_at: str
    metadata: Dict


# ============================================================================
# AI Detection
# ============================================================================

class AIDetector:
    """Detects AI-generated patterns in text"""

    # Common AI fingerprints
    AI_PHRASES = [
        "delve into", "it's important to note", "it's worth noting",
        "landscape of", "navigating the", "realm of", "tapestry of",
        "in conclusion", "in summary", "to summarize",
        "revolutionize", "game-changer", "cutting-edge",
        "leverage", "utilize", "facilitate", "optimize",
        "robust", "comprehensive", "holistic", "seamless",
        "at the end of the day", "moving forward", "going forward",
        "dive deep", "unpack", "break down", "drill down"
    ]

    # AI tends to use these transitions excessively
    AI_TRANSITIONS = [
        "however", "moreover", "furthermore", "additionally",
        "consequently", "therefore", "thus", "hence"
    ]

    # AI patterns: overly structured, predictable
    AI_PATTERNS = [
        r"^\d+\.\s+",  # Numbered lists at start
        r"^-\s+",  # Bullet points
        r":\s*$",  # Colons at end of paragraphs
        r"^\*\*.*\*\*$",  # Bold headings
    ]

    @classmethod
    def calculate_ai_likelihood(cls, text: str) -> float:
        """
        Calculate likelihood that text is AI-generated (0.0-1.0)

        Higher scores = more likely AI-generated
        Target: < 0.25 after cleaning
        """
        score = 0.0
        text_lower = text.lower()

        # Check for AI phrases
        phrase_count = sum(1 for phrase in cls.AI_PHRASES if phrase in text_lower)
        score += min(0.3, phrase_count * 0.03)

        # Check for excessive transitions
        transition_count = sum(1 for trans in cls.AI_TRANSITIONS if text_lower.count(trans) > 2)
        score += min(0.2, transition_count * 0.05)

        # Check for uniform sentence lengths (AI tends to be uniform)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 3:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
                # Low variance = uniform = more likely AI
                if variance < 20:
                    score += 0.2

        # Check for structural patterns
        pattern_count = sum(1 for pattern in cls.AI_PATTERNS
                          if re.search(pattern, text, re.MULTILINE))
        score += min(0.15, pattern_count * 0.05)

        # Check for hedging language (AI tends to hedge)
        hedges = ["might", "could", "perhaps", "possibly", "potentially"]
        hedge_count = sum(text_lower.count(h) for h in hedges)
        if hedge_count > 3:
            score += 0.1

        # Check for repetitive structure (e.g., every paragraph starts same way)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 2:
            first_words = [p.split()[0].lower() for p in paragraphs if p.split()]
            if len(first_words) != len(set(first_words)):
                score += 0.1

        return min(1.0, score)


# ============================================================================
# Linguistic Perturbation
# ============================================================================

class LinguisticPerturber:
    """Applies subtle perturbations to remove AI fingerprints"""

    # Contractions to make text more casual
    CONTRACTIONS = {
        "it is": "it's",
        "that is": "that's",
        "there is": "there's",
        "cannot": "can't",
        "will not": "won't",
        "do not": "don't",
        "does not": "doesn't",
        "did not": "didn't",
        "should not": "shouldn't",
        "would not": "wouldn't",
    }

    # Replacements for overly formal AI language
    CASUALIZE = {
        "utilize": "use",
        "facilitate": "help",
        "implement": "put in place",
        "demonstrate": "show",
        "indicate": "show",
        "numerous": "many",
        "endeavor": "try",
        "commence": "start",
        "terminate": "end",
        "sufficient": "enough",
    }

    @classmethod
    def add_contractions(cls, text: str) -> str:
        """Add contractions to make text more casual"""
        for formal, casual in cls.CONTRACTIONS.items():
            # Only replace some instances (not all) for naturalness
            if random.random() < 0.6:  # 60% chance
                text = text.replace(formal, casual)
        return text

    @classmethod
    def casualize_language(cls, text: str) -> str:
        """Replace overly formal words with casual alternatives"""
        words = text.split()
        result = []
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in cls.CASUALIZE and random.random() < 0.7:
                replacement = cls.CASUALIZE[word_lower]
                # Preserve original capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                punct = ''.join(c for c in word if c in '.,!?;:')
                result.append(replacement + punct)
            else:
                result.append(word)
        return ' '.join(result)

    @classmethod
    def inject_rhythm_irregularity(cls, text: str) -> str:
        """
        Break uniform sentence patterns by occasionally:
        - Merging short sentences
        - Splitting long sentences
        - Adding sentence fragments for emphasis
        """
        sentences = re.split(r'([.!?]+)', text)
        result = []

        i = 0
        while i < len(sentences) - 1:
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else '.'

            if not sentence:
                i += 2
                continue

            word_count = len(sentence.split())

            # Short sentence: occasionally merge with next or make it a fragment
            if word_count < 8 and i + 2 < len(sentences) and random.random() < 0.3:
                next_sentence = sentences[i + 2].strip()
                if next_sentence and len(next_sentence.split()) < 10:
                    # Merge
                    result.append(sentence + punct + ' ' + next_sentence)
                    result.append(sentences[i + 3] if i + 3 < len(sentences) else '.')
                    i += 4
                    continue

            result.append(sentence + punct)
            i += 2

        return ' '.join(result)

    @classmethod
    def remove_ai_markers(cls, text: str) -> str:
        """Remove common AI phrases and markers"""
        # Remove hedging
        text = re.sub(r'\b(it\'s worth noting that|it\'s important to note that)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(in conclusion,?|to summarize,?|in summary,?)\b', '', text, flags=re.IGNORECASE)

        # Remove excessive transition words at start of sentences
        text = re.sub(r'\.\s+(However|Moreover|Furthermore|Additionally),\s+', '. ', text)

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)

        return text.strip()


# ============================================================================
# Voice Baseline
# ============================================================================

class VoiceBaseline:
    """Compares cleaned text against author's voice baseline"""

    def __init__(self, voiceprint_path: Path):
        self.voiceprint_path = voiceprint_path
        self.params = self._load_voiceprint()

    def _load_voiceprint(self) -> Dict:
        """Load VoicePrint from JSON"""
        if not self.voiceprint_path.exists():
            logger.warning(f"VoicePrint not found at {self.voiceprint_path}")
            return {}

        with open(self.voiceprint_path, 'r') as f:
            return json.load(f)

    def calculate_voice_deviation(self, text: str) -> float:
        """
        Calculate deviation from author's voice baseline (0.0-1.0)

        Lower scores = more authentic
        Target: < 0.35
        """
        if not self.params:
            return 0.5  # Default if no baseline

        vp = self.params.get("voice_parameters", {})
        deviation = 0.0

        # Check sentence length distribution
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            target_mean = vp.get("sentence_length", {}).get("mean", 18.5)
            deviation += min(0.3, abs(avg_len - target_mean) / target_mean)

        # Check for common phrases
        common_phrases = vp.get("common_phrases", [])
        text_lower = text.lower()
        phrase_count = sum(1 for phrase in common_phrases if phrase in text_lower)
        if phrase_count == 0 and len(common_phrases) > 0:
            deviation += 0.2

        # Check formality (overly formal = deviation)
        formal_words = ["utilize", "facilitate", "demonstrate", "numerous", "endeavor"]
        formal_count = sum(1 for word in formal_words if word in text_lower)
        if formal_count > 2:
            deviation += 0.15

        return min(1.0, deviation)


# Global instances
voice_baseline = VoiceBaseline(VOICEPRINT_PATH)


# ============================================================================
# Content Cleaning
# ============================================================================

def clean_draft(draft: DraftInput) -> CleanedResponse:
    """Clean draft to remove AI fingerprints"""

    original_content = draft.content

    # Calculate baseline AI likelihood
    ai_likelihood_before = AIDetector.calculate_ai_likelihood(original_content)

    logger.info(f"Cleaning draft: {draft.title}")
    logger.info(f"  AI likelihood before: {ai_likelihood_before:.2f}")

    # Use Claude to intelligently rewrite with anti-AI instructions
    prompt = f"""You are Erebus, an authenticity agent that removes AI fingerprints from text while preserving meaning and voice.

# Task
Rewrite this LinkedIn post to sound MORE HUMAN and LESS AI-generated.

# Original Post
{original_content}

# Anti-AI Instructions

**Remove these AI markers:**
- Phrases like "delve into", "it's worth noting", "landscape of", "tapestry of"
- Excessive transitions ("however", "moreover", "furthermore")
- Overly structured, predictable patterns
- Hedging language ("might", "could", "perhaps", "possibly")
- Corporate jargon ("utilize", "leverage", "facilitate", "optimize")

**Add human irregularity:**
- Use contractions naturally (it's, that's, don't)
- Vary sentence length dramatically (mix very short with longer)
- Occasional sentence fragments for emphasis
- Less formal, more conversational
- Specific, concrete language over abstract
- Direct assertions over hedged statements

**Preserve:**
- Core meaning and key points
- Specific numbers and facts
- Overall structure (hook → body → closing)
- Target word count (~{draft.word_count} words)

Write the cleaned version now (just the post, no commentary):
"""

    try:
        message = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.8,  # Higher temp for more variation
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        cleaned_content = message.content[0].text.strip()

        # Apply additional programmatic perturbations
        cleaned_content = LinguisticPerturber.add_contractions(cleaned_content)
        cleaned_content = LinguisticPerturber.casualize_language(cleaned_content)
        cleaned_content = LinguisticPerturber.remove_ai_markers(cleaned_content)
        cleaned_content = LinguisticPerturber.inject_rhythm_irregularity(cleaned_content)

        # Calculate metrics
        ai_likelihood_after = AIDetector.calculate_ai_likelihood(cleaned_content)
        voice_deviation = voice_baseline.calculate_voice_deviation(cleaned_content)

        # Calculate edit distance (simplified)
        orig_words = original_content.split()
        clean_words = cleaned_content.split()
        edit_distance = abs(len(orig_words) - len(clean_words)) / max(len(orig_words), 1)

        word_count_change = len(clean_words) - len(orig_words)

        # Generate diff summary
        diff_summary = f"Changed {word_count_change:+d} words. "
        if ai_likelihood_after < ai_likelihood_before:
            improvement = (ai_likelihood_before - ai_likelihood_after) / ai_likelihood_before * 100
            diff_summary += f"Reduced AI likelihood by {improvement:.1f}%. "
        diff_summary += f"Voice deviation: {voice_deviation:.2f}"

        # Create response
        cleaned_id = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        response = CleanedResponse(
            cleaned_id=cleaned_id,
            draft_id=draft.draft_id,
            title=draft.title,
            original_content=original_content,
            cleaned_content=cleaned_content,
            diff_summary=diff_summary,
            ai_likelihood_before=ai_likelihood_before,
            ai_likelihood_after=ai_likelihood_after,
            voice_deviation=voice_deviation,
            edit_distance=edit_distance,
            word_count_change=word_count_change,
            created_at=datetime.now().isoformat(),
            metadata={
                **draft.metadata,
                "cleaning_model": "claude-sonnet-4-20250514",
                "perturbations_applied": [
                    "contractions",
                    "casualization",
                    "marker_removal",
                    "rhythm_irregularity"
                ]
            }
        )

        # Save cleaned version
        cleaned_path = CLEANED_DIR / f"{cleaned_id}.json"
        with open(cleaned_path, 'w') as f:
            json.dump(response.dict(), f, indent=2)

        logger.info(f"✓ Cleaned: {cleaned_id}")
        logger.info(f"  AI likelihood after: {ai_likelihood_after:.2f} ({ai_likelihood_before:.2f} → {ai_likelihood_after:.2f})")
        logger.info(f"  Voice deviation: {voice_deviation:.2f}")
        logger.info(f"  Word count: {draft.word_count} → {draft.word_count + word_count_change}")

        return response

    except Exception as e:
        logger.error(f"Error cleaning draft: {e}")
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/healthz")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent": "agent-erebus",
        "version": "1.0.0",
        "voiceprint_loaded": voice_baseline.params is not None
    }


@app.post("/v1/clean", response_model=CleanedResponse)
def clean_content(draft: DraftInput):
    """
    Clean draft to remove AI fingerprints

    Applies linguistic perturbations and reintroduces human irregularity
    """
    return clean_draft(draft)


@app.get("/v1/cleaned/{cleaned_id}")
def get_cleaned(cleaned_id: str):
    """Retrieve specific cleaned content by ID"""
    cleaned_path = CLEANED_DIR / f"{cleaned_id}.json"

    if not cleaned_path.exists():
        raise HTTPException(status_code=404, detail=f"Cleaned content {cleaned_id} not found")

    with open(cleaned_path, 'r') as f:
        return json.load(f)


@app.get("/v1/cleaned")
def list_cleaned(limit: int = 20):
    """List recent cleaned content"""
    cleaned = sorted(CLEANED_DIR.glob("cleaned_*.json"), reverse=True)[:limit]

    return {
        "total": len(list(CLEANED_DIR.glob("cleaned_*.json"))),
        "returned": len(cleaned),
        "cleaned": [
            {
                "cleaned_id": f.stem,
                "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            }
            for f in cleaned
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
