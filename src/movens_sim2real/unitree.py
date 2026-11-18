"""Unitree G1 / Dex3 message adapter. Packet production does not open DDS."""
from math import isfinite


STATE_TOPICS = ("rt/lowstate", "rt/dex3/left/state", "rt/dex3/right/state")
COMMAND_TOPICS = ("rt/lowcmd", "rt/dex3/left/cmd", "rt/dex3/right/cmd")


class G1Packets:
    def __init__(self, body_channels, gains, *, mode_machine, mode_pr):
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_ as new_hand,
            unitree_hg_msg_dds__LowCmd_ as new_body,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        channels = tuple(body_channels)
        if len(channels) != 29 or len(set(channels)) != 29 or not all(type(i) is int and i in range(35) for i in channels):
            raise ValueError("G1 body calibration must name 29 distinct SDK motor slots")
        if mode_pr not in (0, 1) or mode_machine not in range(1, 256):
            raise ValueError("Expected calibrated PR/AB and nonzero G1 machine mode")
        self.kp, self.kd = (tuple(map(float, gains[k])) for k in ("kp", "kd"))
        if any(len(g) != 43 or any(not isfinite(x) or x < 0 for x in g) for g in (self.kp, self.kd)):
            raise ValueError("Expected calibrated nonnegative gains for all 43 actuators")
        self.channels = (channels, tuple(range(7)), tuple(range(7)))
        self.mode_machine, self.mode_pr = mode_machine, mode_pr
        self.state_type = (LowState_, HandState_, HandState_)
        self.new_body, self.new_hand, self.crc = new_body, new_hand, CRC()

    def decode(self, bank, packet, captured_s):
        """Return encoder values on the independently synchronized capture clock."""
        if bank not in (0, 1, 2) or not isinstance(packet, self.state_type[bank]) or not isfinite(captured_s):
            raise ValueError("Invalid SDK bank, message type or capture timestamp")
        if bank == 0 and (self.crc.Crc(packet) != packet.crc or packet.mode_machine != self.mode_machine
                          or packet.mode_pr != self.mode_pr):
            raise ValueError("LowState CRC or calibrated mode mismatch")
        if bank != 0 and any(packet.error):
            raise ValueError(f"Dex3 bank {bank} reports a controller error")
        if len(packet.motor_state) != (35 if bank == 0 else 7):
            raise ValueError("SDK encoder bank size does not match calibration")
        motors = [packet.motor_state[i] for i in self.channels[bank]]
        q, qd = ([float(getattr(m, k)) for m in motors] for k in ("q", "dq"))
        if not all(isfinite(x) for x in (*q, *qd)):
            raise ValueError("Nonfinite SDK encoder measurement")
        return float(captured_s), dict(q=q, qd=qd, body_tick=packet.tick if bank == 0 else None)

    def encode(self, bounded_wire, body_state, *, captured_s, now_s, max_age_s):
        """Pack WireOrder's limited targets; publication alone never proves application."""
        self.decode(0, body_state, captured_s)
        if not (isfinite(max_age_s) and 0 <= now_s-captured_s <= max_age_s <= .06):
            raise ValueError("A fresh calibrated G1 body state is required to form commands")
        targets = tuple(map(float, bounded_wire))
        if len(targets) != 43 or not all(isfinite(x) for x in targets):
            raise ValueError("Expected 43 finite limited wire position targets in radians")
        body, left, right = self.new_body(), self.new_hand(), self.new_hand()
        body.mode_pr, body.mode_machine = self.mode_pr, self.mode_machine
        channel = 0
        for bank, (packet, slots) in enumerate(zip((body, left, right), self.channels)):
            for slot in slots:
                motor = packet.motor_cmd[slot]
                motor.mode = 1 if bank == 0 else (1 << 4) | slot
                motor.q, motor.kp, motor.kd = targets[channel], self.kp[channel], self.kd[channel]
                # No feed-forward torque or velocity for the position servo.
                channel += 1
        body.crc = self.crc.Crc(body)
        return body, left, right
