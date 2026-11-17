//! The interface (`docs/design/DIRECTION.md`): design tokens ([`theme`]), embedded type
//! ([`fonts`]), strings ([`i18n`]), the logomark ([`logo`]), drawn icons ([`icons`]), the
//! components every view is built from ([`components`]), the zones of the window
//! ([`layout`]), the shell around the views ([`shell`], [`palette`]) and the five views
//! ([`views`]). Views draw the state of `app`, `studio` and `agent`; they hold none.

pub mod components;
pub mod fonts;
pub mod i18n;
pub mod icons;
pub mod layout;
pub mod logo;
pub mod palette;
pub mod shell;
pub mod theme;
pub mod views;
pub mod wizard;
