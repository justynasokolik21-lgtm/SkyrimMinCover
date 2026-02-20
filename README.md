# SkyrimMinCover
SkyrimMinCover is a Windows tool that computes the **minimum number of actions**
(crafting potions and/or eating ingredients) needed to discover **all alchemy
effects for every ingredient** in Skyrim.

The solver outputs an **Excel checklist** showing exactly what to craft and which
ingredients are required.

# Instructions
1. Download the .exe file
2. Download one of the example CSVs found under example_csvs
3. Run the .exe
4. Provide the file path to the ingredient CSV, or hit 'browse' and select it.
5. [OPTIONAL] Change the output file path to the location where you want your Excel checklist to be saved. If left unchanged, it will save to the directory that contains the ingredients CSV
6. Press 'Run'
7. Wait for the solver to finish running. If you have a lot of ingredients, this can take a while. As long as the green blob is bouncing, things are fine. The solver quits after 2 minutes of no improvement.
8. When the text updates say "Done," you can close the app.
