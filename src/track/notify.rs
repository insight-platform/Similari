pub trait ChangeNotifier: Default + Clone + Send + Sync {
    fn send(&mut self, id: u64);
}

#[derive(Default, Clone, Debug)]
pub struct NoopNotifier;

impl ChangeNotifier for NoopNotifier {
    fn send(&mut self, _id: u64) {}
}
