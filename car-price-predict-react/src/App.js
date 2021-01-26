import './App.css';
import AppBar from '@material-ui/core/AppBar';
import { makeStyles } from '@material-ui/core/styles';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import MenuIcon from '@material-ui/icons/Menu';
import EvaluateCarPrice from './EvaluateCarPrice/EvaluateCarPrice';

const useStyles = makeStyles((theme) => ({
	root: {
		flexGrow: 1,
	},
	menuButton: {
		marginRight: theme.spacing(2),
	},
	title: {
		flexGrow: 1
	},
}));

function App() {
	const classes = useStyles();

	return (
		<div>
			<div className={classes.root}>
				<AppBar position="static">
					<Toolbar>
						<MenuIcon />
						<Typography variant="h6" className={classes.title}>
							Car Reco
						</Typography>
					</Toolbar>
				</AppBar>
			</div>
			<EvaluateCarPrice/>
		</div>
	);
}

export default App;
