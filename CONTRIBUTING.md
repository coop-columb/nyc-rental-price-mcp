# Contributing to NYC Rental Price Prediction Project

Thank you for considering contributing to our project! This document outlines the best practices for contributing to this repository.

## Git Workflow

### Branch Naming Convention

- `feature/descriptive-name`: For new features
- `bugfix/issue-description`: For bug fixes
- `refactor/component-name`: For code refactoring
- `docs/what-changed`: For documentation updates
- `test/component-name`: For adding or updating tests

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `test`: Adding or updating tests
- `chore`: Changes to the build process or auxiliary tools

Examples:

- `feat(model): add support for XGBoost algorithm`
- `fix(api): resolve null pointer issue in prediction endpoint`
- `docs(readme): update installation instructions`

### Pull Request Process

1. Create a new branch from `main` using the naming convention above
2. Make your changes in small, logical commits
3. Write or update tests as needed
4. Ensure all tests pass locally
5. Push your branch and create a pull request
6. Reference any related issues in the PR description
7. Request a review from at least one team member

## Development Setup

1. Clone the repository
2. Set up your virtual environment
3. Install development dependencies: `pip install -e ".[dev]"`
4. Install pre-commit hooks: `pre-commit install`

## Code Style

This project uses:

- Black for code formatting
- Flake8 for linting
- Type hints for better code quality
- Docstrings in Google format

Run formatting and linting:

```bash
black src tests
flake8 src tests
```

## Testing

Run tests:

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit
pytest tests/integration
pytest tests/performance
```

## Documentation

- All functions and classes should have docstrings
- Update README.md for significant changes
- Consider adding notebook examples for new features

## Release Process

1. Update version in setup.py
2. Create a changelog entry
3. Create a new release on GitHub
4. Tag the release with a version number (e.g., v1.0.0)
