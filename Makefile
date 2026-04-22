# Avoid mixing site-packages: a global PYTHONPATH (e.g. ROS/colcon) can inject a
# different Python’s packages and break native wheels like pydantic_core.
.PHONY: test
test:
	env -u PYTHONPATH uv run pytest tests/ -v
