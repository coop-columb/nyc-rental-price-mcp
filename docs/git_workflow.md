# Git Workflow

This document outlines the Git workflow for the NYC Rental Price Prediction project.

## Branching Strategy

We follow a modified Git Flow workflow:

```bash
main (production-ready code)
├── develop (integration branch)
│   ├── feature/feature-name
│   ├── bugfix/bug-description
│   ├── refactor/component-name
│   ├── docs/documentation-update
│   └── test/test-addition
└── hotfix/urgent-fix (from main)
```

### Branch Types

- `main`: Production-ready code. Protected branch.
- `develop`: Integration branch for features. Protected branch.
- `feature/*`: New features or enhancements.
- `bugfix/*`: Bug fixes.
- `refactor/*`: Code refactoring without changing functionality.
- `docs/*`: Documentation updates.
- `test/*`: Adding or updating tests.
- `hotfix/*`: Urgent fixes for production issues (branches from main).

## Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```structure
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc.)
- `refactor`: Code changes that neither fix bugs nor add features
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Changes to the build process or auxiliary tools
- `ci`: Changes to CI configuration files and scripts

### Examples

- `feat(model): add support for XGBoost algorithm`
- `fix(api): resolve null pointer issue in prediction endpoint`
- `docs(readme): update installation instructions`
- `refactor(preprocessing): simplify data cleaning pipeline`

## Pull Request Process

1. Create a branch from `develop` (or `main` for hotfixes)
2. Make your changes in small, logical commits
3. Write or update tests as needed
4. Ensure all tests pass locally
5. Push your branch and create a pull request to `develop` (or `main` for hotfixes)
6. Reference any related issues in the PR description
7. Request a review from at least one team member
8. Address review comments
9. Once approved, merge using squash merge

## Release Process

1. When `develop` is ready for release, create a release branch `release/vX.Y.Z`
2. Fix any bugs in the release branch
3. When the release is ready, merge to `main` using a merge commit (not squash)
4. Tag the release on `main` with version `vX.Y.Z`
5. Merge `main` back to `develop`

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Git LFS

Large files are managed using Git LFS. See `.gitattributes` for the file types tracked by LFS.

## Best Practices

- Keep commits small and focused on a single change
- Write clear commit messages
- Pull and rebase frequently to avoid conflicts
- Never force push to shared branches (`main`, `develop`, `release/*`)
- Use `git rebase -i` to clean up your branch before creating a PR
- Delete branches after they are merged

## Git Hooks

We use pre-commit hooks to ensure code quality. Install them with:

```bash
pip install pre-commit
pre-commit install
```
