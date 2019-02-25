class Props:
    def __init__(self):
        self._internal = dict()

    def __str__(self):
        result = ''
        for key, value in self._internal.items():
            result += key + ': ' + str(value) + '\n'

        return result


class PipeProps(Props):
    def __init__(self, diameter, cost):
        super().__init__()
        self._internal['diameter'] = diameter
        self._internal['cost'] = cost

    @property
    def diameter(self):
        return self._internal['diameter']

    @diameter.setter
    def diameter(self, value):
        self._internal['diameter'] = value

    @property
    def cost(self):
        return self._internal['cost']

    @cost.setter
    def cost(self, value):
        self._internal['cost'] = value


class NodeProps(Props):
    def __init__(self, position, demand):
        super().__init__()
        self._internal['position'] = position
        self._internal['demand'] = demand

    @property
    def position(self):
        return self._internal['position']

    @position.setter
    def position(self, value):
        self._internal['position'] = value

    @property
    def demand(self):
        return self._internal['demand']

    @demand.setter
    def demand(self, value):
        self._internal['demand'] = value


class EdgeProps(Props):
    def __init__(self, length=1, diameter=0, flow_rate=0, actual_flow=0, No=None, cost=0):
        super().__init__()
        self._internal['length'] = length
        self._internal['diameter'] = diameter
        self._internal['flow_rate'] = flow_rate
        self._internal['actual_flow'] = actual_flow
        self._internal['cost'] = cost
        self._internal['No'] = No

    def __str__(self):
        if self.diameter == 0:
            return ''
        else:
            return super().__str__()

    @property
    def cost(self):
        return self._internal['cost']

    @cost.setter
    def cost(self, value):
        self._internal['cost'] = value

    @property
    def No(self):
        return self._internal['No']

    @No.setter
    def No(self, value):
        self._internal['No'] = value

    @property
    def actual_flow(self):
        return self._internal['actual_flow']

    @actual_flow.setter
    def actual_flow(self, value):
        self._internal['actual_flow'] = value

    @property
    def flow_rate(self):
        return self._internal['flow_rate']

    @flow_rate.setter
    def flow_rate(self, value):
        self._internal['flow_rate'] = value

    @property
    def diameter(self):
        return self._internal['diameter']

    @diameter.setter
    def diameter(self, value):
        self._internal['diameter'] = value

    @property
    def length(self):
        return self._internal['length']

    @length.setter
    def length(self, value):
        self._internal['length'] = value
