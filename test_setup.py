def check_imports():
    packages = [
        "pandas",
        "sklearn",
        "joblib", 
        "matplotlib",
        "scipy",
        "seaborn"
    ]
    
    print("Checking required packages:")
    all_ok = True
    
    for pkg in packages:
        try:
            module = __import__(pkg)
            ver = getattr(module, '__version__', 'unknown')
            print(f"✓ {pkg}: {ver}")
        except ImportError as e:
            print(f"❌ {pkg}: Not installed")
            all_ok = False
        except Exception as e:
            print(f"❌ {pkg}: Error - {str(e)}")
            all_ok = False
    
    return all_ok

if __name__ == '__main__':
    success = check_imports()
    if success:
        print("\n✅ Ambiente configurado corretamente!")
    else:
        print("\n❌ Alguns pacotes estão faltando ou com erro.")