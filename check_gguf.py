import gguf
try:
    print(f"MIMO2 present: {gguf.MODEL_ARCH.MIMO2}")
except AttributeError:
    print("MIMO2 NOT present")
except Exception as e:
    print(f"Error: {e}")
