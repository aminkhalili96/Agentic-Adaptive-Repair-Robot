"""
Test script for the LangGraph repair agent.
"""

import asyncio

from src.agent.graph import RepairAgent, create_agent
from src.agent.tools import DefectInfo


async def _test_agent_impl():
    """Test the repair agent with sample defects."""
    print("=" * 60)
    print("LangGraph Agent Test")
    print("=" * 60)
    
    # Create sample defects
    defects = [
        DefectInfo(
            index=0,
            type="rust",
            position=(0.55, 0.05, 0.26),
            size=4.5,
            confidence=0.92,
        ),
        DefectInfo(
            index=1,
            type="crack",
            position=(0.65, -0.03, 0.26),
            size=2.0,
            confidence=0.78,
        ),
        DefectInfo(
            index=2,
            type="dent",
            position=(0.60, 0.00, 0.26),
            size=6.0,
            confidence=0.85,
        ),
    ]
    
    print(f"\n[1/3] Creating agent...")
    agent = create_agent()  # Uses config (OpenAI or Ollama)
    print("  ✓ Agent ready")
    
    print(f"\n[2/3] Classifying {len(defects)} defects...")
    print("-" * 40)
    
    plans = await agent.run_classification(defects)
    
    print("-" * 40)
    print(f"\n[3/3] Results:")
    print("-" * 40)
    
    for plan in plans:
        print(f"\nDefect {plan.defect_index}: {plan.defect_type.upper()}")
        print(f"  Severity: {plan.severity}")
        print(f"  Strategy: {plan.strategy}")
        print(f"  Tool: {plan.tool}")
        print(f"  Est. Time: {plan.estimated_time}s")
        print(f"  Notes: {plan.notes}")
    
    print("\n" + "=" * 60)
    print("Agent Test COMPLETE!")
    print("=" * 60)


def test_agent():
    asyncio.run(_test_agent_impl())


if __name__ == "__main__":
    test_agent()
