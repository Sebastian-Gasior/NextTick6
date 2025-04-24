from setuptools import setup, find_packages

setup(
    name="ml4t_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "yfinance",
        "scikit-learn",
        "plotly",
        "ta",
    ],
    python_requires=">=3.10",
    author="Sebastian Gasior",
    author_email="sebastian.gasior@example.com",
    description="ML4T Trading Analysis System",
    long_description="ML4T - Machine Learning for Trading Project with advanced features",
    url="https://github.com/Sebastian-Gasior/ml4t_project",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
) 