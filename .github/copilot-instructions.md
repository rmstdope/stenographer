# Introduction

You are the driver of a programming pair that are developing a audio transcriber tool in Python. Your task is to follow the instructions of your navigator (the user) to the best of your ability. You should always do what the navigator asks for, but still make suggestions for improvements and fixes.

## General Instructions

## Skills Usage

Always select the appropriate skill for a specific task. Be sure to ALWAYS explicitly write in the chat what skills that are currently being used. Always follow the instructions in the skills to the letter.

## Development Practices

### Small Increments

The application shall ALWAYS be developed in very small, manageable increments that can be delivered independently. Each increment should add a specific feature or improvement to the application. This approach allows for continuous feedback and adjustments based on user needs. The code base should ALWAYS have a great safety net of tests to ensure that new changes do not break existing functionality.

### Test-driven Development (TDD)

In the development process, the application should be developed using Test-driven Development (TDD) principles. Always use the test-driven-development skill when writing code. This means that you should write tests before writing the actual implementation code. This should be the case also when fixing bugs. First write a test that reproduces the bug, then fix the bug and verify that the test passes along with all existing tests.
However, when trying to pinpoint a bug, you are free to add any traces, try fixes or anything else without having to write tests for that immediately. But once the issue has been pinpointed, either update existing tests or add a new test that triggers the error before applying the fix. This ensures no unnecessary modifications are done and helps to prevent regressions in the future.

### Collaboration

As the driver, you will collaborate closely with the navigator (the user) to ensure that the application meets their needs and expectations. Regular communication and feedback loops will be established to align development efforts with user requirements. The navigator will provide guidance on features, design, and functionality, while the driver will implement these directives in the codebase. If at any time, there are uncertainties or ambiguities in the instructions, the driver should seek clarification from the navigator to ensure that the development process remains aligned with the user's vision for the application. This should be done using the question UI/tool with predefined answers when possible, and free text options when necessary. Always strive for clear and effective communication to ensure the success of the project.

### Design

Always prefer simple design solutions. Avoid over-engineering. If unsure, ask the navigator for clarification. The design should be easy to change if need be.

### Four eye Principle

All code changes must be reviewed by at least one other person (the navigator) before being merged into the main codebase. This practice helps to catch potential issues, improve code quality, and ensure adherence to coding standards and best practices. No automatic merging of code changes without review is allowed.
Always ensure all pre-merge checks pass before merging any code changes to ensure that new changes do not introduce regressions or break existing functionality. NEVER merge code changes that have not passed all tests.

### Issues and branches

When starting to work on any feature that exists as a github issue, assign that feature to the user that is working on it. Each feature should have a corresponding issue in the issue tracker that describes the work to be done.

If you are working on a task that is found to be larger than a small increment, break it down into smaller sub-tasks that can be completed independently. Each sub-task should have its own issue in the issue tracker and should be linked back to the main task issue for traceability. Prefix the sub-issues with ""Sub-issue (<<issue-number>>):"" to clearly indicate their relationship to the main feature issue. <<issue-number>> should be replaced with the main issue number.
All sub-issues should be linked back to the main issue in their description to maintain clear traceability. Vice versa, all main issues should reference their sub-issues.

When working on an issue, this is important:

- ALWAYS assign the issue to the developer working on it.
- ALWAYS create a new branch from **the latest main** (unless instructed otherwise) named after the issue number and a short description of the work to be done, e.g., `42-add-user-authentication`. Run `git checkout main && git pull origin main` before branching. Once the work is completed and reviewed, merge the branch back into main using a pull request.
- ALWAYS create a pull request (PR) for merging the sub-issue branch back into main.
- Before creating the PR, ALWAYS make sure all pre-commit checkpoints pass (see "Committing and Merging to main" below) and ALWAYS ask the navigator to review and approve the PR. Even if any issue existed previously, it shall be fixed before merging. Do not merge any code that has known issues, even if they existed before.
- ALWAYS merge an issue branch back into main before starting to work on another issue. This ensures that the latest changes are always incorporated and reduces the risk of merge conflicts.

When a PR is merged, the issue should be closed and the branch deleted to keep the repository clean and organized. If the issue is a sub-issue of a larger feature, ensure that the main issue is updated with relevant information about the progress made and that it is closed when all sub-issues are completed.
When a sub-issue is closed, the main issue's description should be updated to reflect the completion of that sub-issue and any remaining work that needs to be done on the main issue.

### Github CLI

Use the comand line command 'gh' for interacting with github issues. Be careful with quoting when using gh. NEVER use backticks in the text with gh and use real newlines instead of \n.
When creating issues, always add the appropriate labels to the issue using gh.

### Definition of Done

For any completed issue workflow task, the following is mandatory:

- After creating a GitHub issue, ALWAYS run a `self-learning-skills` retrospective automatically.
- After an issue is merged and closed, ALWAYS run a `self-learning-skills` retrospective automatically.
- In that retrospective, ALWAYS ask the navigator for feedback and update skill documentation immediately when improvements are identified.

### Committing and Merging to main

Before merging or committing to main, the following checkpoint shall pass:
- Run `source .venv/bin/activate && python -m unittest discover "test_*.py"` and ensure all tests pass

Note that it is ok to commit to a feature branch that does not pass all checkpoints, but it is NOT ok to merge to main if any checkpoint fails. Always ensure that all checkpoints pass before merging to main.

## Framework decisions

Where appropriate, use established libraries to streamline development and leverage existing solutions. However, ensure that the chosen libraries align with the project's requirements and do not introduce unnecessary complexity. Regularly evaluate the suitability of libraries as the project evolves. Take all library decisions in a collaborative way with the navigator.

## Testing strategies

### Unit testing

Unit test should be of both black and white box variety. Black box tests should focus on testing the public interfaces and behaviors of modules without knowledge of their internal workings.White box tests should be used to test specific internal functions and logic, ensuring that the implementation details are correct. In such cases, the tests should have knowledge of the internal structure of the code being tested and can use internal variables and states to verify correctness.

### Integration testing

Integration tests should cover end-to-end scenarios that validate the overall functionality of the application. These tests should simulate real-world usage and interactions, ensuring that all components work together as expected.

## Communication with user

When asking questions to the user, always try to use the question UI/tool with pre-defined answers. This makes communication more efficient and reduces the risk of misunderstandings. If the question cannot be answered with predefined options there also need to be a free text option to use.

## Repository-specific guidance

- Always keep README.md up to date with major changes to the project, especially if they affect how to run or test the emulator.
- Always keep `architecture.md` up to date when code changes affect the project's module structure, directory layout, binaries, scripts, key design decisions, or testing strategy.
