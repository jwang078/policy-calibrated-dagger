"""Entry point: `python -m my_scripts.gridworld_dagger_sim`.

Default = launch the GUI. Pass --no_gui (and other flags) to run batch eval.
"""

from __future__ import annotations

from my_scripts.gridworld_dagger_sim.cli import make_parser, resolve_initial_state, run


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.no_gui:
        run(args)
        return
    # GUI mode.
    from my_scripts.gridworld_dagger_sim.gui import launch

    state = resolve_initial_state(args.state)
    launch(state)


if __name__ == "__main__":
    main()
