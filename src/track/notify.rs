pub trait ChangeNotifier: Default + Clone + Sync + Send + 'static {
    fn send(&mut self, id: u64);
}

#[derive(Default, Clone, Debug)]
pub struct NoopNotifier;

impl ChangeNotifier for NoopNotifier {
    fn send(&mut self, _id: u64) {}
}
