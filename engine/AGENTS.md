# Agents

## Testing and results workflow (engine module)

When asked to refresh Solitaire player results:

1. **Run the tests** in the `engine/src/test/java/ai/games/results` package.
2. **Gather the test results** from the logging output for each result suite.
3. **Update the root `README.md` table**:
   - Use the existing columns (**Player**, **AI**, **Games Played**, **Games Won**, **Win %**, **Avg Time/Game**, **Total Time**, **Avg Moves**, **Best Win Streak**, **Notes**).
   - Keep the **AI** column aligned with implementation details (`Search` vs `LLM`).
   - Ensure **Notes** contain clickable links to the implementing classes.
4. **Record when the tests were run** by updating the “last test run” line in the root `README.md`.
5. **Keep engine-specific details** (how to run, profiles, logging) in `engine/README.md`, and avoid duplicating the full results table there.

Use the paths under `engine/src/main/java/ai/games/player/ai/` when linking to player implementations from documentation.
