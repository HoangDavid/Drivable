"""
Main Entry Point
Hand Steering Controller for Roblox
"""

from hand_steering_app import HandSteeringApp


if __name__ == "__main__":
    try:
        app = HandSteeringApp()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

