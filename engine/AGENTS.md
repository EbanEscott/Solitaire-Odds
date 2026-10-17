# Agents

## Testing and results workflow (engine module)

During player development, keep interim results and profiles in run logs (for example, `engine/logs/profiles/`). Do not update the root `README.md` results table or test date for these runs. That table is reserved for settled players.

When asked to publish settled player results or refresh the root results table:

1. **Run the tests** in the `engine/src/test/java/ai/games/results` package.
2. **Gather the test results** from the logging output for each result suite.
3. **Update the root `README.md` table**:
   - Use the existing columns (**Player**, **AI**, **Games Played**, **Games Won**, **Win %**, **Avg Time/Game**, **Total Time**, **Avg Moves**, **Best Win Streak**, **Notes**).
   - Keep the **AI** column aligned with implementation details (`Search` vs `LLM`).
   - Ensure **Notes** contain clickable links to the implementing classes.
4. **Record when the tests were run** by updating the “last test run” line in the root `README.md`.
5. **Keep engine-specific details** (how to run, profiles, logging) in `engine/README.md`, and avoid duplicating the full results table there.

Use the paths under `engine/src/main/java/ai/games/player/ai/` when linking to player implementations from documentation.

## Java documentation

Every class (including nested classes and records), field, constructor, and method must have a descriptive Javadoc-style comment using `/** ... */`, matching the documented search players. Document purpose, behavior, and relevant constraints; use `@param` and `@return` where they clarify the contract.

Every logical code block inside a method must also have an explanatory `/** ... */` comment. Describe the intent of loops, branches, and grouped operations rather than restating individual statements. Apply this style throughout classes you substantially revise, including their tests.

Keep algorithm documentation accurate: explain search depth, scoring, tie-breaking, information boundaries, cycle handling, and termination where relevant. Do not describe local perturbations as restarts or claim a player avoids simulation when it evaluates copied successors.

Do not hard-wrap Markdown prose; let the viewer wrap lines.
