{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Arc2HardwareSimulator:\n",
    "    \"\"\"Models the actual hardware with a fixed probability distribution\n",
    "    \n",
    "    Assumptions:\n",
    "        * This model assumes that probability distribution is fixed and does not\n",
    "          vary across devices in the wafer and across wafers.\n",
    "    \"\"\"\n",
    "    def __init__(self, number_devices: int):\n",
    "        self._number_devices = number_devices\n",
    "        self._state_transitions = [_transition_probability(state,params)\n",
    "                                    for state,params in enumerate(_NON_FAIL_STATE_TPS_PARAMS)]\n",
    "        self._state_transitions.append(lambda x: [0.0, 0.0, 0.0, 1.0])\n",
    "        self._device_state = [random.randrange(NUM_NON_FAIL_STATES) for _ in range(number_devices)]\n",
    "        self._current_device = 0\n",
    "\n",
    "    def get_current_device_state(self):\n",
    "        return self._device_state[self._current_device]\n",
    "\n",
    "    def apply_voltage(self,voltage: np.float32):\n",
    "        _current_state = self._device_state[self._current_device]\n",
    "        _state_transition_probabilities = self._state_transitions[_current_state](voltage)\n",
    "        _next_state = np.random.choice(list(range(NUM_STATES)),p=_state_transition_probabilities)\n",
    "        self._device_state[self._current_device] = _next_state\n",
    "\n",
    "    def move_to_next_device(self) -> bool:\n",
    "        self._current_device += 1\n",
    "        return self._current_device < self._number_devices\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
