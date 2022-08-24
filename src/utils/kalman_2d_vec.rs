pub struct Kalman2dVecState {}

pub struct Kalman2dVec {}

impl Kalman2dVec {
    pub fn new(position_weight: f32, velocity_weight: f32) -> Self {
        todo!()
    }

    pub fn initiate(&self, points: Vec<(f32, f32)>) -> Kalman2dVecState {
        todo!()
    }

    pub fn predict(&self, state: Kalman2dVecState) -> Kalman2dVecState {
        todo!()
    }

    pub fn update(&self, state: Kalman2dVecState, points: Vec<(f32, f32)>) -> Kalman2dVecState {
        todo!()
    }
}
